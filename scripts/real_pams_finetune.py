"""real_pams_finetune.py — Step 1: 真模型 OLMoE-1B-7B PAMS 微调 + A 口径前后measurement

frozen declaration: 训练 forward, 第 11 个, 超出 6 实验解冻范围 (①-⑥), 需用户授权解冻。
purpose: verification PAMS (Precision-Aware Margin Shaping) 在真模型的外部效度。
  from-scratch probe 三轮已证主命题强稳健 (4/4 seed 降翻转 93.4% + loss 反降 11%);
  本轮在 OLMoE-1B-7B-0125 真模型上微调仅 gate weight, 测 A 口径翻转率前后变化 + loss + d_t。

design:
  - loading OLMoE-0125 bf16 GPU, set_norm_topk, model.train()
  - 冻结仅 *.mlp.gate.weight (~2.1M params, 16×131k), expert/attn/emb 全冻
  - PAMS hook: 每层 block.gate (OlmoeTopKRouter) forward hook 捕获 out[0]=router_logits
    (带梯度), 算 d_t=z_sorted[k-1]-z_sorted[k], pams_reg=relu(ε-d_t).sum(), 累加侧信道 list
  - 微调循环: forward → ce_loss + λ·Σpams → backward → AdamW(gate only) step
  - measurement before/after (A 口径, C′ 口径):
    * bf16: 当前模型 GPU forward 抓 router_logits
    * fp32: 独立loading CPU 模型, 覆盖 gate weight = 当前 gate weight.float(), forward 抓 logits
    * flip_A = bf16 top-k vs fp32 top-k 翻转率 (L4/L8/L12)
    * d_t 分布 (bf16 logits) + loss (CE held-out)
  - verdict: 翻转降>20% + |loss_delta|<10% baseline => PAMS_REAL_REDUCES_FLIP_LOSS_HELD (E)

口径: OLMoE-1B-7B-0125 非 0924 canonical。A 口径=C′ 口径 (0125 baseline L8=5.94%, Step0 标定)。
  不用 gradient checkpointing (丢弃 hook logits 计算图破坏 PAMS 梯度流)。
output: results/real_pams_finetune.json
环境: MTEAR_MODEL (默认 allenai/OLMoE-1B-7B-0125), PAMS_STEPS (默认 300),
  PAMS_LAMBDA (默认 0.01), PAMS_EPS (默认 0.0116, Step0 L8 标定),
  PAMS_LAYERS (默认 4,8,12), PAMS_N_WIN_MEASURE (默认 20)
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
OUT_PATH = OUT_DIR + f"/real_pams_finetune{_out_tag}.json"


def topk_set_masks(logits, k):
    return calib.topk_set_masks(logits, k)


def pct(t, ps):
    return calib.pct(t, ps)


# PAMS regularization侧信道: 每层 hook 累加 pams_reg (带梯度), 每 step 前 clear
_PAMS_REGS = []


def install_pams_hooks(blocks, eps, k):
    # 每层 block.gate forward hook: 捕获 out[0] router_logits, 算 pams_reg 累加
    def hook(mod, _inp, out):
        logits = out[0]  # OlmoeTopKRouter 返回 (router_logits, scores, indices), 带梯度
        zf = logits.float()
        z_sorted = zf.sort(dim=-1, descending=True).values
        d_t = z_sorted[:, k - 1] - z_sorted[:, k]
        _PAMS_REGS.append(F.relu(eps - d_t).sum())

    handles = [blk.gate.register_forward_hook(hook) for _, blk, _ in blocks]
    return handles


def measure(src_model, ids, starts, layer_idxs, device, k, model_id, eps):
    # A 口径measurement: bf16 (当前 src_model GPU) vs fp32 (独立loading CPU, gate weight 覆盖)
    src_model.eval()
    blocks = mtp.discover_moe_blocks(src_model)
    # bf16 forward 抓 router_logits + d_t
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
            # CE loss (next-token, bf16 logits → float)
            ce = F.cross_entropy(out.logits[0, :-1].float(), wid[0, 1:])
            ce_losses.append(float(ce))
    for h in handles:
        h.remove()
    log_bf = {li: torch.cat(v).cpu() for li, v in grabbed_bf.items()}
    dt_cat = {li: torch.cat(v).cpu() for li, v in dt_list.items()}

    # fp32 独立loading CPU, 覆盖 gate weight = src gate weight.float()
    print(f"[measure] loading fp32 CPU model for A 口径 ...")
    fp_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
    ).eval()
    mtp.set_norm_topk(fp_model, "auto")
    blocks_fp = mtp.discover_moe_blocks(fp_model)
    for li in layer_idxs:
        blocks_fp[li][1].gate.weight.data = blocks[li][1].gate.weight.detach().float().cpu()
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
        m0 = topk_set_masks(log_bf[li], k).cpu()
        m1 = topk_set_masks(log_fp[li], k).cpu()
        flip_a = float((m0 != m1).any(-1).float().mean())
        n_tok = int(log_bf[li].shape[0])
        n_flip = int((m0 != m1).any(-1).sum())
        dt = dt_cat[li]
        dt_pct = pct(dt, [1, 5, 25, 50, 75, 95, 99])
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
    pams_lambda = float(os.environ.get("PAMS_LAMBDA", "0.01"))
    pams_eps = float(os.environ.get("PAMS_EPS", "0.0116"))
    layers_req = [int(x) for x in os.environ.get("PAMS_LAYERS", "4,8,12").split(",")]
    n_win_measure = int(os.environ.get("PAMS_N_WIN_MEASURE", "20"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[finetune] model={model_id} steps={steps} λ={pams_lambda} ε={pams_eps} "
          f"layers={layers_req} device={device} k={calib.K}")

    tok = AutoTokenizer.from_pretrained(model_id)
    print(f"[finetune] loading bf16 model ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device, low_cpu_mem_usage=True
    )
    mtp.set_norm_topk(model, "auto")
    blocks = mtp.discover_moe_blocks(model)
    print(f"[finetune] {len(blocks)} moe blocks")

    # 冻结仅 gate weight
    n_train = 0
    for n, p in model.named_parameters():
        is_gate = n.endswith("mlp.gate.weight")
        p.requires_grad = is_gate
        if is_gate:
            n_train += p.numel()
    print(f"[finetune] trainable params: {n_train} (gate only)")
    gate_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(gate_params, lr=1e-4, weight_decay=0.0)
    print(f"[finetune] optimizer: AdamW lr=1e-4 on {len(gate_params)} gate tensors")

    ids = p3.load_corpus_ids(tok)
    print(f"[finetune] corpus ids len={len(ids)}")
    # measurement窗 (held-out 用前段, 训练用后段, 避免泄露)
    measure_starts = list(range(0, len(ids) - calib.WINDOW + 1, calib.STEP))[:n_win_measure]
    train_starts = list(range(n_win_measure * calib.STEP + calib.WINDOW,
                              len(ids) - calib.WINDOW + 1, calib.STEP))

    # PAMS hooks (训练时累加regularization)
    handles = install_pams_hooks(blocks, pams_eps, calib.K)

    # === before measurement ===
    print(f"[finetune] === BEFORE measurement ===")
    before = measure(model, ids, measure_starts, layers_req, device, calib.K, model_id, pams_eps)

    # === 微调循环 ===
    print(f"[finetune] === 微调 {steps} steps ===")
    model.train()
    loss_log = []
    pams_log = []
    for step in range(steps):
        start = train_starts[step % len(train_starts)]
        _PAMS_REGS.clear()
        wid = ids[start:start + calib.WINDOW].unsqueeze(0).to(device)
        enc = {"input_ids": wid, "attention_mask": torch.ones_like(wid)}
        out = model(**enc)
        ce = F.cross_entropy(out.logits[0, :-1].float(), wid[0, 1:])
        pams_total = sum(_PAMS_REGS) if _PAMS_REGS else torch.zeros((), device=device)
        total = ce + pams_lambda * pams_total
        total.backward()
        torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()
        loss_log.append(float(ce.detach()))
        pams_log.append(float(pams_total.detach()))
        if step % 50 == 0 or step == steps - 1:
            print(f"[finetune] step {step} ce={float(ce):.4f} pams_reg={float(pams_total):.4f}")
    for h in handles:
        h.remove()

    # === after measurement ===
    print(f"[finetune] === AFTER measurement ===")
    after = measure(model, ids, measure_starts, layers_req, device, calib.K, model_id, pams_eps)

    # verdict
    flip_before_l8 = next(l["flip_A"] for l in before["per_layer"] if l["layer_idx"] == 8)
    flip_after_l8 = next(l["flip_A"] for l in after["per_layer"] if l["layer_idx"] == 8)
    flip_red = (flip_before_l8 - flip_after_l8) / flip_before_l8 if flip_before_l8 > 0 else 0.0
    loss_delta = after["loss"] - before["loss"]
    loss_rel = abs(loss_delta) / before["loss"] if before["loss"] > 0 else 0.0
    if flip_red > 0.2 and loss_rel < 0.10:
        verdict = "PAMS_REAL_REDUCES_FLIP_LOSS_HELD"
    elif flip_red > 0.2:
        verdict = "PAMS_REAL_REDUCES_FLIP_LOSS_INCREASED"
    else:
        verdict = "PAMS_REAL_FLIP_NOT_REDUCED"
    print(f"[finetune] L8 flip: {flip_before_l8:.4f}→{flip_after_l8:.4f} red={flip_red:.3f} | "
          f"loss: {before['loss']:.4f}→{after['loss']:.4f} delta={loss_delta:+.4f}")
    print(f"[finetune] verdict={verdict}")

    res = {
        "config": {"model": model_id, "steps": steps, "pams_lambda": pams_lambda,
                   "pams_eps": pams_eps, "k": calib.K, "window": calib.WINDOW,
                   "layers": layers_req, "n_win_measure": n_win_measure, "lr": 1e-4,
                   "trainable_params": n_train},
        "before": before, "after": after,
        "loss_curve_tail": loss_log[-10:],
        "pams_reg_curve_tail": pams_log[-10:],
        "flip_reduction_L8": flip_red, "loss_delta": loss_delta, "loss_rel": loss_rel,
        "verdict": verdict, "evidence_tier": "E",
        "note": ("真模型 OLMoE-1B-7B-0125 PAMS 微调仅 gate weight。A 口径=C′ 口径 (bf16 vs fp32 全 forward)。"
                 "0125 非 0924 canonical。SFT 级局部微调非 fully-converged。第 11 个训练 forward, 用户授权解冻。"),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[finetune] 写入 {OUT_PATH}")


if __name__ == "__main__":
    main()