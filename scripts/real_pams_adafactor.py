"""real_pams_adafactor.py — Adafactor 全parameter PAMS 微调 (外部效度verification, 最接近 probe 全parameter)

frozen declaration: 训练 forward, 第 16 个, 超出 6 实验解冻范围, 用户授权 (走 Adafactor 全parameter方案)。
背景: 仅训 gate 三 λ (0.05/0.3/1.0) 失败 + gate+L8 expert 失败, confirm PAMS 在真模型仅训 router/
  局部 expert 无法全局push up margin (d_t 中位始终unchanged)。probe 成功因全parameter训练 (gate+expert+attn
  协同, PAMS 能push up margin 10.95×)。本轮 Adafactor 全parameter (7B 全训) + gradient checkpointing
  减激活, 最接近 probe 全parameter设置, verification大模型全parameter协同能否push up margin 降翻转。

关键技术 (绕过 PAMS hook/checkpoint 冲突):
  - gradient checkpointing 丢弃 DecoderLayer 中间激活, PAMS 若用 hook 抓 gate logits (带梯度)
    则梯度图指向被丢弃激活, backward 失败。
  - 解法: PAMS regularization不用 hook 抓 logits, 改 forward 后用 captured mlp input h (detach 成常量 fp32)
    + gate.weight 重算 gate logits, 算 pams_reg。h detach 后 pams_reg 计算图只有 gate.weight 叶子,
    梯度独立于 checkpoint 通过 gate.weight 回传。checkpoint backward 重算时 hook 再触发 append h,
    但 compute_pams_reg 在 forward 后取 _BLOCK_H[:n_blocks] (forward 时的, backward 之前)。
  - total = ce + λ·pams_reg, total.backward(): ce 分支经 checkpoint 重算 (算全parameter梯度),
    pams_reg 分支经重算 logits (h 常量, gate.weight 叶子, 独立回传)。gate.weight.grad = 两者之和。

design:
  - loading OLMoE-0125 bf16 GPU, gradient_checkpointing_enable(use_reentrant=False), model.train()
  - 全parameter解冻 (7B), torch.optim.Adafactor (factored 内存高效, state ~1GB 远小于 AdamW 28GB),
    relative_step=False, lr=PAMS_LR (默认 1e-4), grad_clip 1.0
  - 每steps: forward (checkpoint, hook 抓每层 mlp input h detach fp32) → ce → compute_pams_reg
    (用 _BLOCK_H[:16] + gate.weight 重算) → total = ce + λ·pams_reg → backward → step
  - before/after A 口径measurement (measure_full: fp32 baseline=当前模型转 fp32 覆盖所有权重, 非仅 gate,
    否则全parameter微调后 fp32 baseline污染)
  - verdict: L8 flip 降>20% + L8 dt_med push up>20% => PAMS_FULL_REDUCES_FLIP_MARGIN_PUSHED

口径: OLMoE-1B-7B-0125 非 0924 canonical。全parameter SFT 级 300 steps局部微调非 fully-converged。
  gradient checkpointing 开, PAMS 用重算非 hook 绕过冲突。Adafactor 非 AdamW。
  A 口径=C′ 口径 (bf16 vs fp32 全 forward), fp32 baseline=当前模型转 fp32 (全parameter口径)。
显存: 模型 bf16 14GB + grad 14GB + Adafactor state ~1.5GB + checkpoint 激活 ~0.5GB ≈ 30GB (32GB 紧).
  OOM 降级: 冻 attn+emb 省 grad 1.2GB, 或减 window。
output: results/real_pams_adafactor{PAMS_TAG}.json
环境: PAMS_STEPS (默认 300), PAMS_LAMBDA (默认 0.05), PAMS_EPS (默认 0.0116),
  PAMS_LR (默认 1e-4), PAMS_N_WIN_MEASURE (默认 20), MTEAR_MODEL (默认 allenai/OLMoE-1B-7B-0125)
"""
import os
import sys
import json

import torch
import torch.nn.functional as F

PROBE_DIR = os.environ.get("MTEAR_PROBE_DIR", "/path/to/project/moe-tearing-main")
PHASE3_DIR = "/path/to/project/moe-tear-experiments/scripts"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROBE_DIR)
sys.path.insert(0, PHASE3_DIR)
sys.path.insert(0, SCRIPT_DIR)

import moe_tear_probe as mtp
import phase3_experimentA as p3
import real_pams_eps_calib as calib
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = SCRIPT_DIR + "/../results"
os.makedirs(OUT_DIR, exist_ok=True)
_out_tag = os.environ.get("PAMS_TAG", "")
OUT_PATH = OUT_DIR + f"/real_pams_adafactor{_out_tag}.json"

# 侧信道: 每层 mlp input h (detach fp32), forward 时 hook append, backward checkpoint 重算再 append
# compute_pams_reg 取 _BLOCK_H[:n_blocks] (forward 时的, backward 之前)
_BLOCK_H = []


def install_h_hooks(blocks):
    # hook on OlmoeSparseMoeBlock 抓 inp[0] = post_attention_layernorm output = gate input h
    def hook(mod, inp, _out):
        h = inp[0].detach()  # [B, T, H] bf16 常量
        _BLOCK_H.append(h.reshape(-1, h.shape[-1]).float())  # [B*T, H] fp32
    handles = [blk.register_forward_hook(hook) for _, blk, _ in blocks]
    return handles


def compute_pams_reg(blocks, eps, k, device):
    # 用 captured h (detach fp32) + gate.weight 重算 gate logits 算 pams_reg
    # h 是常量, gate.weight 带梯度, logits 梯度回传到 gate.weight (经 w.float() cast)
    n_blocks = len(blocks)
    h_list = _BLOCK_H[:n_blocks]  # forward 时抓的 (前 n_blocks 个)
    if len(h_list) < n_blocks:
        return torch.zeros((), device=device)
    regs = []
    for (name, blk, _), h in zip(blocks, h_list):
        w = blk.gate.weight  # bf16 [E, H], requires_grad=True
        logits = F.linear(h, w.float())  # [N, E] fp32, 带梯度到 w (经 float cast 回传)
        z_sorted = logits.sort(dim=-1, descending=True).values
        d_t = z_sorted[:, k - 1] - z_sorted[:, k]
        regs.append(F.relu(eps - d_t).sum())
    return sum(regs)


def measure_full(src_model, ids, starts, layer_idxs, device, k, model_id, eps):
    # A 口径measurement: bf16 当前模型 GPU forward vs fp32 当前模型 (所有权重转 fp32) CPU forward
    # 全parameter微调后必须覆盖所有权重, 否则 fp32 baseline污染 (仅覆盖 gate 假设只有 gate 变)
    src_model.eval()
    blocks = mtp.discover_moe_blocks(src_model)
    grabbed_bf = {li: [] for li in layer_idxs}
    dt_list = {li: [] for li in layer_idxs}

    def mk(li):
        def hook(mod, _inp, out):
            logits = out[0].detach().reshape(-1, out[0].shape[-1]).float()
            grabbed_bf[li].append(logits)
            z_sorted = logits.sort(dim=-1, descending=True).values
            dt_list[li].append(z_sorted[:, k - 1] - z_sorted[:, k])
        return hook

    handles = [blocks[li][1].gate.register_forward_hook(mk(li)) for li in layer_idxs]
    ce_losses = []
    with torch.no_grad():
        for start in starts:
            wid = ids[start:start + calib.WINDOW].unsqueeze(0).to(device)
            enc = {"input_ids": wid, "attention_mask": torch.ones_like(wid)}
            out = src_model(**enc)
            ce = F.cross_entropy(out.logits[0, :-1].float(), wid[0, 1:])
            ce_losses.append(float(ce))
    for h in handles:
        h.remove()
    log_bf = {li: torch.cat(v).cpu() for li, v in grabbed_bf.items()}
    dt_cat = {li: torch.cat(v).cpu() for li, v in dt_list.items()}

    # fp32 当前模型: loading原始 fp32 CPU baseline, 用 src_model state_dict 覆盖所有权重
    print(f"[measure] loading fp32 CPU base + override all weights from src ...")
    fp_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
    ).eval()
    mtp.set_norm_topk(fp_model, "auto")
    sd = {k: v.detach().float().cpu() for k, v in src_model.state_dict().items()}
    fp_model.load_state_dict(sd)
    del sd
    blocks_fp = mtp.discover_moe_blocks(fp_model)
    grabbed_fp = {li: [] for li in layer_idxs}

    def mkf(li):
        def hook(mod, _inp, out):
            logits = out[0].detach().reshape(-1, out[0].shape[-1]).float()
            grabbed_fp[li].append(logits)
        return hook

    handles = [blocks_fp[li][1].gate.register_forward_hook(mkf(li)) for li in layer_idxs]
    with torch.no_grad():
        for start in starts:
            wid = ids[start:start + calib.WINDOW].unsqueeze(0)
            enc = {"input_ids": wid, "attention_mask": torch.ones_like(wid)}
            fp_model(**enc)
    for h in handles:
        h.remove()
    log_fp = {li: torch.cat(v) for li, v in grabbed_fp.items()}
    del fp_model

    per_layer = []
    for li in layer_idxs:
        m0 = calib.topk_set_masks(log_bf[li], k).cpu()
        m1 = calib.topk_set_masks(log_fp[li], k).cpu()
        flip_a = float((m0 != m1).any(-1).float().mean())
        n_tok = int(log_bf[li].shape[0])
        n_flip = int((m0 != m1).any(-1).sum())
        dt = dt_cat[li]
        dt_pct = calib.pct(dt, [1, 5, 25, 50, 75, 95, 99])
        frac_below = float((dt < eps).float().mean())
        per_layer.append({
            "layer_idx": li, "flip_A": flip_a, "n_tokens": n_tok, "n_flip": n_flip,
            "dt_pct": dt_pct, "frac_dt_below_eps": frac_below,
        })
        print(f"[measure] L{li}: flip_A={flip_a:.6f} ({n_flip}/{n_tok}) dt_med={dt_pct[50]:.4f} "
              f"dt_1%={dt_pct[1]:.4f} frac<ε={frac_below:.4f}")
    loss_mean = sum(ce_losses) / len(ce_losses) if ce_losses else None
    src_model.train()
    return {"per_layer": per_layer, "loss": loss_mean}


def main():
    model_id = os.environ.get("MTEAR_MODEL", "allenai/OLMoE-1B-7B-0125")
    steps = int(os.environ.get("PAMS_STEPS", "300"))
    pams_lambda = float(os.environ.get("PAMS_LAMBDA", "0.05"))
    pams_eps = float(os.environ.get("PAMS_EPS", "0.0116"))
    lr = float(os.environ.get("PAMS_LR", "1e-5"))
    layers_req = [4, 8, 12]
    n_win_measure = int(os.environ.get("PAMS_N_WIN_MEASURE", "20"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[adafactor] model={model_id} steps={steps} λ={pams_lambda} ε={pams_eps} "
          f"lr={lr} layers={layers_req} device={device} k={calib.K}")

    tok = AutoTokenizer.from_pretrained(model_id)
    print(f"[adafactor] loading bf16 model ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device, low_cpu_mem_usage=True
    )
    mtp.set_norm_topk(model, "auto")
    # gradient checkpointing 减激活 (全parameter训练必须)
    # transformers 5.x: GradientCheckpointingLayer 自管理 reentrant, 无 use_reentrant parameter
    model.gradient_checkpointing_enable()
    model.train()
    blocks = mtp.discover_moe_blocks(model)
    n_blocks = len(blocks)
    print(f"[adafactor] {n_blocks} moe blocks, checkpoint enabled")

    # 全parameter解冻 (保持 bf16, 用 lr 3×补偿 bf16 更新阈值)
    n_train = 0
    gate_params = []
    other_params = []
    for n, p in model.named_parameters():
        p.requires_grad = True
        n_train += p.numel()
        if "gate.weight" in n:
            gate_params.append(p)
        else:
            other_params.append(p)
    print(f"[adafactor] trainable params: {n_train} (全parameter, gate lr 3×)", flush=True)

    optimizer = torch.optim.Adafactor(
        [
            {"params": other_params, "lr": lr},
            {"params": gate_params, "lr": lr * 3},
        ],
        weight_decay=0.0, eps=(1e-3, 1e-3),
    )
    print(f"[adafactor] optimizer: Adafactor lr={lr} (gate 3×={lr*3}) factored eps=(1e-3,1e-3)")

    ids = p3.load_corpus_ids(tok)
    print(f"[adafactor] corpus ids len={len(ids)}")
    measure_starts = list(range(0, len(ids) - calib.WINDOW + 1, calib.STEP))[:n_win_measure]
    train_starts = list(range(n_win_measure * calib.STEP + calib.WINDOW,
                              len(ids) - calib.WINDOW + 1, calib.STEP))

    # h hooks (训练时抓 mlp input h detach fp32, 用于 PAMS 重算)
    handles = install_h_hooks(blocks)

    # === before measurement ===
    print(f"[adafactor] === BEFORE measurement ===")
    model.eval()
    before = measure_full(model, ids, measure_starts, layers_req, device, calib.K, model_id, pams_eps)
    model.train()

    # === 微调循环 ===
    grad_clip = float(os.environ.get("PAMS_GRAD_CLIP", "0.5"))
    print(f"[adafactor] === 微调 {steps} steps (全parameter Adafactor) grad_clip={grad_clip} ===")
    loss_log = []
    pams_log = []
    for step in range(steps):
        start = train_starts[step % len(train_starts)]
        _BLOCK_H.clear()
        wid = ids[start:start + calib.WINDOW].unsqueeze(0).to(device)
        enc = {"input_ids": wid, "attention_mask": torch.ones_like(wid)}
        out = model(**enc)
        ce = F.cross_entropy(out.logits[0, :-1].float(), wid[0, 1:])
        pams_total = compute_pams_reg(blocks, pams_eps, calib.K, device)
        total = ce + pams_lambda * pams_total
        if torch.isnan(total) or torch.isinf(total):
            # 诊断: 哪些parameter grad 已爆
            n_nan_grad = 0
            n_inf_grad = 0
            for n, p in model.named_parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any():
                        n_nan_grad += 1
                    if torch.isinf(p.grad).any():
                        n_inf_grad += 1
            print(f"[adafactor] step {step} total NaN/Inf! ce={float(ce.detach()):.4f} "
                  f"pams_reg={float(pams_total.detach()):.4f} nan_grad_params={n_nan_grad} inf_grad_params={n_inf_grad}")
            break
        total.backward()
        # 诊断 grad nan/inf (backward 后)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            n_nan_grad = 0
            n_inf_grad = 0
            for n, p in model.named_parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any():
                        n_nan_grad += 1
                    if torch.isinf(p.grad).any():
                        n_inf_grad += 1
            print(f"[adafactor] step {step} grad_norm={float(grad_norm):.4f} NaN/Inf after backward! "
                  f"nan_grad_params={n_nan_grad} inf_grad_params={n_inf_grad}")
            break
        optimizer.step()
        optimizer.zero_grad()
        loss_log.append(float(ce.detach()))
        pams_log.append(float(pams_total.detach()))
        if step % 50 == 0 or step == steps - 1:
            print(f"[adafactor] step {step} ce={float(ce.detach()):.4f} "
                  f"pams_reg={float(pams_total.detach()):.4f} "
                  f"grad_norm={float(grad_norm):.3f} "
                  f"mem={torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
    for h in handles:
        h.remove()

    # === after measurement ===
    print(f"[adafactor] === AFTER measurement ===")
    model.eval()
    after = measure_full(model, ids, measure_starts, layers_req, device, calib.K, model_id, pams_eps)

    # verdict (看 L8)
    L8 = 8
    flip_b = next(l["flip_A"] for l in before["per_layer"] if l["layer_idx"] == L8)
    flip_a = next(l["flip_A"] for l in after["per_layer"] if l["layer_idx"] == L8)
    flip_red = (flip_b - flip_a) / flip_b if flip_b > 0 else 0.0
    dt_b = next(l["dt_pct"][50] for l in before["per_layer"] if l["layer_idx"] == L8)
    dt_a = next(l["dt_pct"][50] for l in after["per_layer"] if l["layer_idx"] == L8)
    dt_push = (dt_a - dt_b) / dt_b if dt_b > 0 else 0.0
    loss_delta = after["loss"] - before["loss"]
    loss_rel = abs(loss_delta) / before["loss"] if before["loss"] > 0 else 0.0
    if flip_red > 0.2 and dt_push > 0.2:
        verdict = "PAMS_FULL_REDUCES_FLIP_MARGIN_PUSHED"
    elif flip_red > 0.2 and loss_rel < 0.10:
        verdict = "PAMS_FULL_REDUCES_FLIP_LOSS_HELD"
    elif flip_red > 0.2:
        verdict = "PAMS_FULL_REDUCES_FLIP_LOSS_INCREASED"
    elif dt_push > 0.2:
        verdict = "MARGIN_PUSHED_BUT_FLIP_NOT_REDUCED"
    else:
        verdict = "FULL_PARAM_STILL_NOT_WORKING"
    print(f"[adafactor] L8 flip: {flip_b:.4f}→{flip_a:.4f} red={flip_red:.3f} | "
          f"dt_med: {dt_b:.4f}→{dt_a:.4f} push={dt_push:.3f} | "
          f"loss: {before['loss']:.4f}→{after['loss']:.4f} delta={loss_delta:+.4f}")
    print(f"[adafactor] verdict={verdict}")

    res = {
        "config": {"model": model_id, "steps": steps, "pams_lambda": pams_lambda,
                   "pams_eps": pams_eps, "k": calib.K, "window": calib.WINDOW,
                   "layers": layers_req, "n_win_measure": n_win_measure, "lr": lr,
                   "optimizer": "Adafactor", "trainable_params": n_train,
                   "scope": "全parameter 7B + gradient_checkpointing + PAMS 重算(非hook)"},
        "before": before, "after": after,
        "loss_curve_tail": loss_log[-10:],
        "pams_reg_curve_tail": pams_log[-10:],
        "flip_reduction_L8": flip_red, "dt_push_L8": dt_push,
        "L8_dt_med_before": dt_b, "L8_dt_med_after": dt_a,
        "loss_delta": loss_delta, "loss_rel": loss_rel,
        "verdict": verdict, "evidence_tier": "E",
        "note": ("Adafactor 全parameter PAMS 微调 (7B 全训), 最接近 probe 全parameter设置。"
                 "gradient checkpointing 开, PAMS 用 forward 后重算 (captured h detach + gate.weight) "
                 "绕过 hook/checkpoint 冲突。A 口径 fp32 baseline=当前模型转 fp32 覆盖所有权重。"
                 "OLMoE-1B-7B-0125 非 0924 canonical。全parameter SFT 级 300 steps非 fully-converged。"
                 "第 16 个训练 forward, 用户授权 (走 Adafactor 全parameter方案)。"),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[adafactor] 写入 {OUT_PATH}")


if __name__ == "__main__":
    main()