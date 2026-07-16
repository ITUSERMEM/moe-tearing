"""real_pams_eps_calib.py — Step 0: 真模型 OLMoE gate |z| 量级标定 + ε_bf16 推定

frozen declaration: 零训练 forward, 仅推理 capture + 单层 gate 数值比较, 不微调, 不算第 11 个训练 forward。
purpose: PAMS 阈值 ε 作用在 logit 空间 d_t, 须取 logit 空间 bf16 舍入误差上界 2δ_z + η。
  probe (moe_train_probe) ε=0.008 对应 |z|~1.1; OLMoE 真模型 |z| 量级未知, 必须实测。
  δ_z ≈ 2^-8 · |z|_median (bf16 mantissa 8 bit 舍入误差上界), ε = 2δ_z + 0.5δ_z (η=0.5δ_z 安全余量)。

measurement:
  - L4/L8/L12 三层 gate logits |z| 分布分位 (1/5/25/50/75/95/99%) + δ_z + ε 推定
  - d_t = z_sorted[k-1]-z_sorted[k] 分布 (中位/1%分位/frac<ε), 看 baseline margin 与 ε 关系
  - B 口径翻转率 baseline (单层 gate bf16 vs fp32 top-k 翻转), confirm真模型 B 口径量级 (control preflight B=0.25%)

口径: OLMoE-1B-7B-0125, 非论文 canonical 0924。B 口径单层 gate ≠ C′ 全模型 bf16 翻转 5.2%。
output: results/real_pams_eps_calibration.json
环境: MTEAR_MODEL (默认 allenai/OLMoE-1B-7B-0125), CALIB_N_WIN (默认 50), CALIB_LAYERS (默认 4,8,12)
"""
import os
import sys
import json

import torch
import torch.nn.functional as F

PROBE_DIR = os.environ.get("MTEAR_PROBE_DIR", "/path/to/project/moe-tearing-main")
PHASE3_DIR = "/path/to/project/moe-tear-experiments/scripts"
sys.path.insert(0, PROBE_DIR)
sys.path.insert(0, PHASE3_DIR)

import moe_tear_probe as mtp
import phase3_experimentA as p3
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../results"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = OUT_DIR + "/real_pams_eps_calibration.json"
WINDOW = 128
STEP = 64
K = 8  # OLMoE-1B-7B top-k


def pct(t, ps):
    # t: 1D fp32 tensor; ps: 百分位列表; 返回 {p: float}
    t = t.flatten().float()
    out = {}
    for p in ps:
        out[p] = float(torch.quantile(t, p / 100.0))
    return out


def topk_set_masks(logits, k):
    # 与 mtp._topk_set_masks 等价: topk indices -> one-hot 布尔
    idx = logits.topk(k, dim=-1).indices
    masks = torch.zeros_like(logits, dtype=torch.bool)
    masks.scatter_(1, idx, True)
    return masks


def capture_layer_h(model, ids, blocks_by_name, starts, device):
    # 一次 forward hook 多层 block 抓 pre-router input h (bf16), 返回 {name: [N, H] fp32}
    grabbed = {name: [] for name in blocks_by_name}

    def mk(name):
        def hook(_mod, inp, _out):
            h = inp[0].detach()
            grabbed[name].append(h.reshape(-1, h.shape[-1]))
        return hook

    handles = [mod.register_forward_hook(mk(name)) for name, mod in blocks_by_name.items()]
    try:
        with torch.no_grad():
            for start in starts:
                wid = ids[start:start + WINDOW].unsqueeze(0).to(device)
                enc = {"input_ids": wid, "attention_mask": torch.ones_like(wid)}
                model(**enc)
    finally:
        for h in handles:
            h.remove()
    return {name: torch.cat(chunks, dim=0).float() for name, chunks in grabbed.items()}


def capture_gate_logits(model_id, ids, starts, layer_idxs, device, dtype, k):
    # loading dtype 模型, forward, hook block.gate 抓 router_logits (out[0]), 返回 {li: [N, E] fp32}
    # A 口径: bf16 全 forward gate logits vs fp32 全 forward gate logits (上游累积量化差异)
    print(f"[calib-A] loading {model_id} dtype={dtype} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=device, low_cpu_mem_usage=True
    ).eval()
    mtp.set_norm_topk(model, "auto")
    blocks = mtp.discover_moe_blocks(model)
    grabbed = {li: [] for li in layer_idxs}

    def mk(li):
        def hook(mod, _inp, out):
            logits = out[0]  # OlmoeTopKRouter 返回 (router_logits, scores, indices)
            grabbed[li].append(logits.detach().reshape(-1, logits.shape[-1]).float())
        return hook

    handles = [blocks[li][1].gate.register_forward_hook(mk(li)) for li in layer_idxs]
    try:
        with torch.no_grad():
            for start in starts:
                wid = ids[start:start + WINDOW].unsqueeze(0).to(device)
                enc = {"input_ids": wid, "attention_mask": torch.ones_like(wid)}
                model(**enc)
    finally:
        for h in handles:
            h.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {li: torch.cat(chunks, dim=0) for li, chunks in grabbed.items()}


def analyze_layer(name, mod, h_fp32, k, eps):
    # gate logits fp32: F.linear(h, gate.weight) — 与 mtp._gate_logits 5.x 分支一致
    w_fp32 = mod.gate.weight.detach().float()
    logits = F.linear(h_fp32, w_fp32)  # [N, E] fp32
    absz = logits.abs().flatten()
    z_pct = pct(absz, [1, 5, 25, 50, 75, 95, 99])
    delta_z = (2.0 ** -8) * z_pct[50]  # |z|_median 舍入误差上界
    eps_calc = 2.0 * delta_z + 0.5 * delta_z  # 2δ_z + η
    # d_t 分布 (raw logits, k 与 k+1 的 margin)
    z_sorted = logits.sort(dim=-1, descending=True).values
    d_t = z_sorted[:, k - 1] - z_sorted[:, k]
    dt_pct = pct(d_t, [1, 5, 25, 50, 75, 95, 99])
    frac_below = float((d_t < eps).float().mean())
    # B 口径翻转率: 单层 gate fp32 vs bf16-then-fp32
    w_bf16 = mod.gate.weight.detach().to(torch.bfloat16).float()
    logits_bf = F.linear(h_fp32, w_bf16)
    m0 = topk_set_masks(logits, k)
    m1 = topk_set_masks(logits_bf, k)
    flip_b = float((m0 != m1).any(-1).float().mean())
    return {
        "layer": name,
        "n_tokens": int(h_fp32.shape[0]),
        "z_abs_pct": z_pct,
        "delta_z": delta_z,
        "eps_calc": eps_calc,
        "dt_pct": dt_pct,
        "frac_dt_below_eps": frac_below,
        "flip_B_baseline": flip_b,
        "gate_weight_shape": list(mod.gate.weight.shape),
    }


def main():
    model_id = os.environ.get("MTEAR_MODEL", "allenai/OLMoE-1B-7B-0125")
    n_win = int(os.environ.get("CALIB_N_WIN", "50"))
    layers_req = [int(x) for x in os.environ.get("CALIB_LAYERS", "4,8,12").split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[calib] model={model_id} device={device} n_win={n_win} layers={layers_req} k={K}")

    tok = AutoTokenizer.from_pretrained(model_id)
    print(f"[calib] loading model bf16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device, low_cpu_mem_usage=True
    ).eval()
    mtp.set_norm_topk(model, "auto")

    blocks = mtp.discover_moe_blocks(model)
    print(f"[calib] discovered {len(blocks)} moe blocks")
    # blocks[i] 对应 layer i (model.model.layers[i].mlp)
    blocks_by_name = {name: mod for name, mod, _ in blocks}
    # 选 layers_req 对应的 block (按出现顺序索引)
    blocks_sel = {}
    for li in layers_req:
        if li < len(blocks):
            name, mod, n_exp = blocks[li]
            blocks_sel[name] = mod
            print(f"[calib] layer {li} -> {name} n_exp={n_exp}")

    ids = p3.load_corpus_ids(tok)
    print(f"[calib] corpus ids len={len(ids)}")
    starts = list(range(0, len(ids) - WINDOW + 1, STEP))[:n_win]
    print(f"[calib] capturing h over {len(starts)} windows ...")
    H = capture_layer_h(model, ids, blocks_sel, starts, device)

    eps_primary = None
    per_layer = []
    for name, mod in blocks_sel.items():
        # 先用临时 eps=inf 算 z/d_t, 再回填 eps
        info = analyze_layer(name, mod, H[name], K, float("inf"))
        # 用本层 eps_calc 重算 frac_below (更准)
        eps_l = info["eps_calc"]
        d_t = F.linear(H[name], mod.gate.weight.detach().float()).sort(-1, descending=True).values
        d_t = d_t[:, K - 1] - d_t[:, K]
        info["frac_dt_below_eps_calc"] = float((d_t < eps_l).float().mean())
        per_layer.append(info)
        print(f"[calib] {name}: |z|_med={info['z_abs_pct'][50]:.4f} δ_z={info['delta_z']:.6f} "
              f"ε={eps_l:.6f} dt_med={info['dt_pct'][50]:.4f} dt_1%={info['dt_pct'][1]:.4f} "
              f"frac<ε={info['frac_dt_below_eps_calc']:.4f} flip_B={info['flip_B_baseline']:.6f}")
        if "layer.8." in name or name.endswith(".8.mlp"):
            eps_primary = eps_l

    # 释放 bf16 模型, 测 A 口径 (bf16 vs fp32 全 forward routing翻转, C′ 口径)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[calib] === A 口径: bf16 vs fp32 全 forward routing翻转 (C′ 口径) ===")
    log_bf = capture_gate_logits(model_id, ids, starts, layers_req, device, torch.bfloat16, K)
    # fp32 全模型 28GB 显存紧, 用 CPU forward (110GB RAM 够), 稳但慢
    log_fp = capture_gate_logits(model_id, ids, starts, layers_req, "cpu", torch.float32, K)
    a_oral = []
    for li in layers_req:
        if li not in log_bf or li not in log_fp:
            continue
        m0 = topk_set_masks(log_bf[li], K).cpu()
        m1 = topk_set_masks(log_fp[li], K).cpu()
        flip_a = float((m0 != m1).any(-1).float().mean())
        n_tok = int(log_bf[li].shape[0])
        n_flip = int((m0 != m1).any(-1).sum())
        a_oral.append({"layer_idx": li, "flip_A": flip_a, "n_tokens": n_tok, "n_flip": n_flip})
        print(f"[calib-A] layer {li}: flip_A={flip_a:.6f} ({n_flip}/{n_tok})")

    res = {
        "config": {"model": model_id, "k": K, "window": WINDOW, "step": STEP,
                   "n_win": len(starts), "layers": layers_req, "device": device},
        "per_layer": per_layer,
        "eps_primary_L8": eps_primary,
        "flip_A_per_layer": a_oral,
        "note": ("ε_bf16 = 2δ_z + 0.5δ_z, δ_z=2^-8·|z|_median (logit 空间 bf16 舍入误差上界)。"
                 "B 口径单层 gate bf16 vs fp32 权重量化翻转 (baseline 实测见 per_layer.flip_B_baseline)。"
                 "A 口径 bf16 vs fp32 全 forward routing翻转 = C′ 口径 (OLMoE-0125 baseline ~5.2%)。"
                 "A 口径才是 PAMS 微调前后有信号的measurement; B 口径若 baseline=0 则单层权重量化不足以翻 top-k。"
                 "OLMoE-1B-7B-0125 非 0924 canonical。零训练 forward, 不算解冻。"),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[calib] eps_primary_L8={eps_primary}")
    print(f"[calib] 写入 {OUT_PATH}")


if __name__ == "__main__":
    main()