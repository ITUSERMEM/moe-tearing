"""实验①: local soft gate cost 50 seed排除 Type II.

purpose: 把 local soft gate 的 loss cost 从"5-seed benefit + 2-run 零结果"升级为
50 seed的真实 bootstrap CI, 要么检出真实代价、要么把 Type II 概率压到可发表水平.
对应 limitation #5 (cost undetected at ~3% power) + mock review 6→accept 路径 (b).

design (paired per-seed, 复用 phase9):
  - hard baseline = 原生 OLMoE forward (不 patch, 与 phase9 line 138 一致)
  - local_soft = patch local_soft_gate (phase9.local_soft_gate + make_forward, m0=0.05)
  - 每 seed 用 np.random.default_rng(seed) 采 N_WIN 个独立窗口起始位置
  - hard 与 local_soft 用相同窗口 (paired), cost_seed = median(local) - median(hard)
  - 50 个 cost_seed -> bootstrap 1000 次 median CI (seed 2026)
  - patch 一次 local_soft 跑完所有 seed 的 local, 再 restore 跑 hard (省 patch 次数)

算力: OLMoE-1B-7B bf16 14GB, 32GB GPU 够. 50 seed × 2 phase × 10 win = 1000 forward,
~20-40 min GPU.

env:
  MTEAR_MODEL (default allenai/OLMoE-1B-7B-0924)
  MTEAR_K (default 8)  MTEAR_M0 (default 0.05)
  N_SEEDS (default 50)  N_WIN_PER_SEED (default 10)
  MTEAR_PROBE_DIR / MTEAR_SCRIPTS_DIR / MTEAR_OUT

产出: results/exp1_cost_50seeds.json, evidence_tier=E (扩样本零结果或检出),
含 costs[] / bootstrap CI / contains_zero / power 声明.
"""
import sys, os, json, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE_DIR = os.environ.get("MTEAR_PROBE_DIR", _HERE)
sys.path.insert(0, _PROBE_DIR)
import moe_tear_probe as mtp
sys.path.insert(0, os.environ.get("MTEAR_SCRIPTS_DIR", _HERE))
import phase9_experimentD_local as p9
import phase3_experimentA as p3

MODEL = os.environ.get("MTEAR_MODEL", "allenai/OLMoE-1B-7B-0924")
K = int(os.environ.get("MTEAR_K", "8"))
M0 = float(os.environ.get("MTEAR_M0", "0.05"))
N_SEEDS = int(os.environ.get("N_SEEDS", "50"))
N_WIN = int(os.environ.get("N_WIN_PER_SEED", "10"))
WINDOW = 128
OUT = Path(os.environ.get("MTEAR_OUT", os.path.join(_HERE, "..", "results")))
BOOT_SEED = 2026
N_BOOT = 1000


def run_loss(model, ids, starts):
    """per-token CE over given window starts. starts: list[int]."""
    losses = []
    dev = next(model.parameters()).device
    with torch.no_grad():
        for st in starts:
            wid = ids[st:st + WINDOW].unsqueeze(0).to(dev)
            enc = {"input_ids": wid, "attention_mask": torch.ones_like(wid)}
            out = model(**enc)
            l = F.cross_entropy(out.logits[0, :-1].float(), wid[0, 1:],
                                reduction="none").cpu().numpy()
            losses.append(l)
    return np.concatenate(losses)


def starts_per_seed(n_seeds, max_start):
    """每 seed 一组独立窗口起始位置 (replace=False). 返回 list[np.ndarray]."""
    return [np.random.default_rng(s).choice(max_start, size=N_WIN, replace=False)
            for s in range(n_seeds)]


def boot_ci(costs):
    costs = np.asarray(costs, dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    boots = np.array([np.median(rng.choice(costs, size=len(costs), replace=True))
                      for _ in range(N_BOOT)])
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {"cost_median": float(np.median(costs)),
            "cost_mean": float(np.mean(costs)),
            "cost_ci_95": [lo, hi],
            "contains_zero": bool(lo <= 0 <= hi),
            "n_seeds": int(len(costs)),
            "n_boot": N_BOOT,
            "min_cost": float(costs.min()), "max_cost": float(costs.max()),
            "n_positive": int((costs > 0).sum()), "n_negative": int((costs < 0).sum())}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda"
    dtype = torch.bfloat16
    print(f"[①] cost 50seeds: model={MODEL} k={K} m0={M0} seeds={N_SEEDS} "
          f"win/seed={N_WIN} (OLMoE, bf16, GPU)", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    print("[①] loading OLMoE bf16 到 GPU (14GB)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=dtype, device_map=device, low_cpu_mem_usage=True).eval()
    mtp.set_norm_topk(model, "auto")
    ids = p3.load_corpus_ids(tok)
    max_start = int(ids.shape[0] - WINDOW)
    print(f"[①] corpus ids={ids.shape}, max_start={max_start}", flush=True)

    starts_list = starts_per_seed(N_SEEDS, max_start)

    # Phase 1: local_soft (patch 一次, 跑所有 seed)
    print(f"[①] patch local_soft (m0={M0}), 跑 {N_SEEDS} seed 的 local loss...", flush=True)
    _, orig = p9.patch(model, p9.local_soft_gate, K, M0)
    local_medians = []
    for s, starts in enumerate(starts_list):
        loss = run_loss(model, ids, starts)
        local_medians.append(float(np.median(loss)))
        if (s + 1) % 10 == 0:
            print(f"  local seed {s+1}/{N_SEEDS}: median={local_medians[-1]:.4f}", flush=True)
    p9.restore(orig)

    # Phase 2: hard (原生, 不 patch), 相同窗口
    print(f"[①] 原生 hard forward, 跑 {N_SEEDS} seed 的 hard loss (paired 窗口)...", flush=True)
    hard_medians = []
    for s, starts in enumerate(starts_list):
        loss = run_loss(model, ids, starts)
        hard_medians.append(float(np.median(loss)))
        if (s + 1) % 10 == 0:
            print(f"  hard seed {s+1}/{N_SEEDS}: median={hard_medians[-1]:.4f}", flush=True)

    costs = np.array(local_medians) - np.array(hard_medians)
    ci = boot_ci(costs)

    print("\n" + "=" * 70)
    print(f"cost (local_soft - hard) over {N_SEEDS} seeds:")
    print(f"  median = {ci['cost_median']:+.5f}")
    print(f"  mean   = {ci['cost_mean']:+.5f}")
    print(f"  95% CI = [{ci['cost_ci_95'][0]:+.5f}, {ci['cost_ci_95'][1]:+.5f}]")
    print(f"  contains_zero = {ci['contains_zero']}")
    print(f"  n_positive={ci['n_positive']}  n_negative={ci['n_negative']}  "
          f"(of {N_SEEDS})")
    print(f"  range = [{ci['min_cost']:+.5f}, {ci['max_cost']:+.5f}]")
    print("=" * 70)

    if ci["contains_zero"]:
        verdict = "ZERO_RESULT_AT_POWER"
        print(f"\n[verdict] 50 seed CI 仍含零 -> cost 在 50 seed 功率下未检出 (Type II 概率降低).")
        print(f"  limitation #5 升级: 从 2-run 零结果 -> 50-seed 零结果, 功率声明更强.")
    else:
        sign = "正" if ci["cost_median"] > 0 else "负"
        verdict = f"DETECTED_{sign}"
        print(f"\n[verdict] 50 seed CI 不含零 -> 检出真实 cost ({sign}), {ci['cost_median']:+.5f}.")
        print(f"  limitation #5 升级: 零结果 -> 有功效measurement, 报告真实代价.")

    out = {
        "experiment": "exp1_cost_50seeds",
        "model": MODEL, "k": K, "m0": M0,
        "n_seeds": N_SEEDS, "n_win_per_seed": N_WIN, "window": WINDOW,
        "hard_medians": hard_medians, "local_medians": local_medians,
        "costs": costs.tolist(),
        "bootstrap": ci,
        "verdict": verdict,
        "evidence_tier": "E",
        "power_declaration": (f"50-seed bootstrap CI over paired per-seed "
                              f"median loss difference (local_soft - hard). "
                              f"CI {'includes' if ci['contains_zero'] else 'excludes'} zero."),
        "note": ("local_soft_gate from phase9 (m0=0.05, margin 渗入, 不归一化, "
                 "与 hard norm_topk_prob=False 一致). hard=原生 forward 不 patch. "
                 "paired: 同 seed 同窗口. 对应 limitation #5 + mock review 6→accept 路径 (b)."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    outpath = OUT / "exp1_cost_50seeds.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\n[done] {outpath}  verdict={verdict}", flush=True)


if __name__ == "__main__":
    main()