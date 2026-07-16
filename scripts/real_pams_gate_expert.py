"""real_pams_gate_expert.py — verification机制: 解冻 L8 gate + L8 expert (单层全parameter近似 probe)

frozen declaration: 训练 forward, 第 15 个, 超出 6 实验解冻范围, 用户授权 (改训 gate+expert verification机制)。
背景: 仅训 gate 三 λ (0.05/0.3/1.0) 都未push up d_t 中位 + 翻转升 3×, confirm机制问题。
  probe 成功因全parameter训练 (gate+expert协同); 真模型仅训 gate expert冻结, gate 全 token 共享
  无法选择性push upboundary margin。本轮解冻 L8 gate + L8 expert verificationexpert协同是否关键。

design:
  - 解冻: model.layers.8.mlp.gate.weight + L8 experts.gate_up_proj + L8 experts.down_proj
    (~405M expert + 131k gate), 其余全冻
  - PAMS hook 只 L8 (block.gate forward hook, out[0] router_logits)
  - 微调 300 steps, 测 L8 翻转/d_t/loss (A 口径) + control L4/L12 (未解冻, 应几乎unchanged)
  - 若 L8 d_t push up + 翻转降 => expert协同是关键, PAMS 需训 expert
  - 若 L8 仍不push up => 更深机制问题

口径: OLMoE-1B-7B-0125 非 0924。L8 expert ~405M params, AdamW state ~4.8GB, 总显存 ~20GB, 不用 checkpoint。
output: results/real_pams_gate_expert{PAMS_TAG}.json
环境: PAMS_STEPS (默认 300), PAMS_LAMBDA (默认 0.05), PAMS_EPS (默认 0.0116)
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
import real_pams_finetune as rpf
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = SCRIPT_DIR + "/../results"
os.makedirs(OUT_DIR, exist_ok=True)
_out_tag = os.environ.get("PAMS_TAG", "")
OUT_PATH = OUT_DIR + f"/real_pams_gate_expert{_out_tag}.json"
L8 = 8


def main():
    model_id = os.environ.get("MTEAR_MODEL", "allenai/OLMoE-1B-7B-0125")
    steps = int(os.environ.get("PAMS_STEPS", "300"))
    pams_lambda = float(os.environ.get("PAMS_LAMBDA", "0.05"))
    pams_eps = float(os.environ.get("PAMS_EPS", "0.0116"))
    layers_req = [4, 8, 12]
    n_win_measure = int(os.environ.get("PAMS_N_WIN_MEASURE", "20"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gate_expert] model={model_id} steps={steps} λ={pams_lambda} ε={pams_eps} L8 gate+expert device={device}")

    tok = AutoTokenizer.from_pretrained(model_id)
    print(f"[gate_expert] loading bf16 model ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device, low_cpu_mem_usage=True
    )
    mtp.set_norm_topk(model, "auto")
    blocks = mtp.discover_moe_blocks(model)
    print(f"[gate_expert] {len(blocks)} moe blocks")

    # 解冻: L8 gate + L8 expert (gate_up_proj + down_proj), 其余全冻
    n_train = 0
    for n, p in model.named_parameters():
        is_l8_gate = (n == "model.layers.8.mlp.gate.weight")
        is_l8_expert = ("model.layers.8.mlp.experts.gate_up_proj" in n
                        or "model.layers.8.mlp.experts.down_proj" in n)
        p.requires_grad = is_l8_gate or is_l8_expert
        if p.requires_grad:
            n_train += p.numel()
    print(f"[gate_expert] trainable params: {n_train} (L8 gate + L8 expert)")
    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(train_params, lr=1e-4, weight_decay=0.0)

    ids = p3.load_corpus_ids(tok)
    measure_starts = list(range(0, len(ids) - calib.WINDOW + 1, calib.STEP))[:n_win_measure]
    train_starts = list(range(n_win_measure * calib.STEP + calib.WINDOW,
                              len(ids) - calib.WINDOW + 1, calib.STEP))

    # PAMS hook 只 L8
    l8_block = blocks[L8][1]
    _PAMS_REGS = []

    def hook(mod, _inp, out):
        logits = out[0]
        zf = logits.float()
        z_sorted = zf.sort(dim=-1, descending=True).values
        d_t = z_sorted[:, calib.K - 1] - z_sorted[:, calib.K]
        _PAMS_REGS.append(F.relu(pams_eps - d_t).sum())

    handle = l8_block.gate.register_forward_hook(hook)

    # before
    print(f"[gate_expert] === BEFORE ===")
    before = rpf.measure(model, ids, measure_starts, layers_req, device, calib.K, model_id, pams_eps)

    # 微调
    print(f"[gate_expert] === 微调 {steps} steps (L8 gate+expert) ===")
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
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()
        loss_log.append(float(ce.detach()))
        pams_log.append(float(pams_total.detach()))
        if step % 50 == 0 or step == steps - 1:
            print(f"[gate_expert] step {step} ce={float(ce.detach()):.4f} pams_reg={float(pams_total.detach()):.4f}")
    handle.remove()

    # after
    print(f"[gate_expert] === AFTER ===")
    after = rpf.measure(model, ids, measure_starts, layers_req, device, calib.K, model_id, pams_eps)

    # verdict (看 L8)
    flip_b_l8 = next(l["flip_A"] for l in before["per_layer"] if l["layer_idx"] == L8)
    flip_a_l8 = next(l["flip_A"] for l in after["per_layer"] if l["layer_idx"] == L8)
    flip_red = (flip_b_l8 - flip_a_l8) / flip_b_l8 if flip_b_l8 > 0 else 0.0
    loss_delta = after["loss"] - before["loss"]
    dt_b = next(l["dt_pct"][50] for l in before["per_layer"] if l["layer_idx"] == L8)
    dt_a = next(l["dt_pct"][50] for l in after["per_layer"] if l["layer_idx"] == L8)
    if flip_red > 0.2 and dt_a > dt_b * 1.2:
        verdict = "EXPERT_COWORK_REDUCES_FLIP_MARGIN_PUSHED"
    elif flip_red > 0.2:
        verdict = "EXPERT_COWORK_REDUCES_FLIP_MARGIN_NOT_PUSHED"
    elif dt_a > dt_b * 1.2:
        verdict = "MARGIN_PUSHED_BUT_FLIP_NOT_REDUCED"
    else:
        verdict = "STILL_NOT_WORKING"
    print(f"[gate_expert] L8 flip: {flip_b_l8:.4f}→{flip_a_l8:.4f} red={flip_red:.3f} | "
          f"dt_med: {dt_b:.4f}→{dt_a:.4f} | loss: {before['loss']:.4f}→{after['loss']:.4f}")
    print(f"[gate_expert] verdict={verdict}")

    res = {
        "config": {"model": model_id, "steps": steps, "pams_lambda": pams_lambda,
                   "pams_eps": pams_eps, "k": calib.K, "window": calib.WINDOW,
                   "layers": layers_req, "trainable_params": n_train,
                   "scope": "L8 gate + L8 expert"},
        "before": before, "after": after,
        "loss_curve_tail": loss_log[-10:],
        "pams_reg_curve_tail": pams_log[-10:],
        "L8_flip_reduction": flip_red, "L8_dt_med_before": dt_b, "L8_dt_med_after": dt_a,
        "loss_delta": loss_delta, "verdict": verdict, "evidence_tier": "E",
        "note": ("机制verification: 解冻 L8 gate + L8 expert (单层全parameter近似 probe), verificationexpert协同是否让 PAMS push up margin 降翻转。"
                 "OLMoE-1B-7B-0125 非 0924。第 15 个训练 forward, 用户授权。"),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[gate_expert] 写入 {OUT_PATH}")


if __name__ == "__main__":
    main()