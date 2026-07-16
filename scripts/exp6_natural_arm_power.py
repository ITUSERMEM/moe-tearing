"""实验⑥: 自然arm power 扩样本 (r≈0.02 零结果的功效声明).

purpose: 论文 limitation "自然arm E, power pending" —— 实验C (phase6) 自然观察arm得
dist_norm vs log_loss 偏相关 r≈0.02 (零结果: boundary邻近度与预测质量无相关), 但
N_WINDOWS=40 (~5000 token) 功效不足, mock review 质询 "在更大 N 下能否检出真实
效应, 还是样本太少". 本实验扩 N_WINDOWS -> ~50000 token, cluster bootstrap by
window 算偏相关 r 的 95% CI, 上界即"在该 N 下能检出的最小效应量", 做有力零结果
声明.

design (复用 phase6 数据收集 + analyze 偏相关口径, 不改原仓库):
  - Step 1: patch phase6 N_WINDOWS=LARGE (default 400) + OUT, 跑 phase6.main()
            -> experimentC_raw_large.json (~50000 token, 400 窗 × 127)
            phase6 max_length=70000 够 (400×64=25600 < 70000)
  - Step 2: 读 raw, 复现 analyze_experimentC 偏相关口径:
            控制 log_freq + hidden_norm + position, 残差相关 r_partial(dist_norm, log_loss)
  - Step 3: cluster bootstrap by window (窗口为 cluster, 重采样窗口, 非独立 token):
            1000 次重采样窗口 -> 重算 partial r -> 95% CI
            (token 级 bootstrap 忽略窗口内自相关, cluster by window 更诚实)
  - Step 4: power 声明: CI 上界 = |r| 的检出下限. 若 |上界| < 阈值 (0.05) ->
            在该 N 下能排除 |r|>=0.05 的效应, 零结果有力 (从 E 零结果升 E+power 声明).

算力: OLMoE-1B-7B 0125 bf16 14GB, 32GB GPU 够. 400 窗 × 全层 hook forward ~15-30
min. bootstrap 1000 × partial lstsq (4 变量, ~50000 token) ~2-5 min CPU.

env:
  MTEAR_MODEL (default allenai/OLMoE-1B-7B-0125)
  MTEAR_K (default 8)  N_WINDOWS_LARGE (default 400)
  N_BOOT (default 1000)  BOOT_SEED (default 2026)
  MTEAR_PROBE_DIR / MTEAR_SCRIPTS_DIR / MTEAR_OUT

产出: results/exp6_natural_arm_power.json + experimentC_raw_large.json, evidence_tier=E.
CI 上界 < 0.05 -> 零结果有力 (power 声明); CI 上界 >= 0.05 -> 功效仍不足, 报告降级.
"""
import sys, os, json
from pathlib import Path
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE_DIR = os.environ.get("MTEAR_PROBE_DIR", _HERE)
sys.path.insert(0, _PROBE_DIR)
import moe_tear_probe as mtp
sys.path.insert(0, os.environ.get("MTEAR_SCRIPTS_DIR", _HERE))
import phase6_experimentC as p6

MODEL = os.environ.get("MTEAR_MODEL", "allenai/OLMoE-1B-7B-0125")
K = int(os.environ.get("MTEAR_K", "8"))
N_WIN_LARGE = int(os.environ.get("N_WINDOWS_LARGE", "400"))
N_BOOT = int(os.environ.get("N_BOOT", "1000"))
BOOT_SEED = int(os.environ.get("BOOT_SEED", "2026"))
OUT = Path(os.environ.get("MTEAR_OUT", os.path.join(_HERE, "..", "results")))
RAW_NAME = "experimentC_raw_large.json"
POWER_THRESHOLD = 0.05  # |r| 检出下限阈值


def run_phase6_large_n():
    """patch phase6 大 N_WINDOWS, 跑数据收集, 返 raw dict."""
    p6.MODEL = MODEL
    p6.K = K
    p6.N_WINDOWS = N_WIN_LARGE
    p6.OUT = OUT
    # phase6 main() 硬写 experimentC_raw.json, 跑后改名
    print(f"[⑥] Step1: phase6 N_WINDOWS={N_WIN_LARGE} (model={MODEL})", flush=True)
    p6.main()
    raw_path = OUT / "experimentC_raw.json"
    raw = json.loads(raw_path.read_text())
    # 改名为 large 避免覆盖原 N=40 基准
    (OUT / RAW_NAME).write_text(json.dumps(raw))
    return raw


def std(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def partial_r(dist_norm, log_loss, log_freq, log_norm, position):
    """复现 analyze_experimentC 偏相关: 控制 freq+norm+position 的残差相关."""
    n = len(log_loss)
    con = np.column_stack([std(log_freq), std(log_norm), position / 128.0, np.ones(n)])
    res_loss = std(log_loss) - con @ np.linalg.lstsq(con, std(log_loss), rcond=None)[0]
    res_dist = std(dist_norm) - con @ np.linalg.lstsq(con, std(dist_norm), rcond=None)[0]
    return float(np.corrcoef(res_dist, res_loss)[0, 1])


def cluster_bootstrap_partial_r(raw, n_boot, seed):
    """cluster bootstrap by window: 重采样窗口 (非 token) 重算 partial r.
    每 window 127 token, window_id = arange(N)//127."""
    loss = np.array(raw["loss"])
    min_dist = np.array(raw["min_dist"])
    hidden_norm = np.array(raw["hidden_norm"])
    freq = np.array(raw["freq"])
    position = np.array(raw["position"])
    dist_norm = np.array(raw["dist_norm"])
    valid = (loss > 0) & np.isfinite(min_dist) & (hidden_norm > 0) & np.isfinite(dist_norm)
    idx_valid = np.where(valid)[0]
    loss, min_dist, hidden_norm, freq, position, dist_norm = [
        x[valid] for x in [loss, min_dist, hidden_norm, freq, position, dist_norm]]
    log_loss = np.log(loss + 1e-12)
    log_freq = np.log(freq + 1)
    log_norm = np.log(hidden_norm)

    r_point = partial_r(dist_norm, log_loss, log_freq, log_norm, position)
    n_valid = len(loss)
    win_size = 127
    # window_id 基于原始 valid 前的全局 token 索引; valid 后需重建
    # phase6 每 window 127 token (T-1), valid 过滤后窗口boundary可能错位,
    # 用 valid 前的全局索引 // 127 作 window_id, 再随 valid 取
    global_idx = idx_valid
    window_ids = global_idx // win_size
    unique_wins = np.unique(window_ids)

    rng = np.random.default_rng(seed)
    boot_rs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_wins, size=len(unique_wins), replace=True)
        idx_chunks = [np.where(window_ids == w)[0] for w in sampled]
        idx = np.concatenate(idx_chunks)
        if len(idx) < 50:
            continue
        boot_rs.append(partial_r(dist_norm[idx], log_loss[idx], log_freq[idx],
                                 log_norm[idx], position[idx]))
    boot_rs = np.array(boot_rs)
    ci_lo, ci_hi = float(np.percentile(boot_rs, 2.5)), float(np.percentile(boot_rs, 97.5))
    return {
        "r_point": r_point, "n_valid": n_valid, "n_boot": len(boot_rs),
        "ci_95": [ci_lo, ci_hi], "boot_rs": boot_rs.tolist(),
        "abs_ci_upper": max(abs(ci_lo), abs(ci_hi)),
        "n_windows": int(len(unique_wins)),
    }


def main():
    print(f"[⑥] 自然arm power: model={MODEL} k={K} N_WIN={N_WIN_LARGE} "
          f"n_boot={N_BOOT}", flush=True)
    raw = run_phase6_large_n()
    print(f"[⑥] Step1 done: N={raw['N']} token, n_windows={raw['n_windows']}", flush=True)

    print(f"[⑥] Step2+3: 偏相关 r + cluster bootstrap by window ({N_BOOT}x)...", flush=True)
    boot = cluster_bootstrap_partial_r(raw, N_BOOT, BOOT_SEED)

    print("\n" + "=" * 70)
    print(f"自然arm偏相关 (dist_norm vs log_loss, 控制 freq+norm+position):")
    print(f"  N_valid = {boot['n_valid']} token  ({boot['n_windows']} windows)")
    print(f"  r_point = {boot['r_point']:+.4f}  (论文 N=40: r≈0.02)")
    print(f"  95% CI  = [{boot['ci_95'][0]:+.4f}, {boot['ci_95'][1]:+.4f}]")
    print(f"  |CI 上界| = {boot['abs_ci_upper']:.4f}  (检出下限, 阈值 {POWER_THRESHOLD})")
    print("=" * 70)

    if boot["abs_ci_upper"] < POWER_THRESHOLD:
        verdict = "POWERFUL_NULL"
        print(f"\n[verdict] |CI 上界|={boot['abs_ci_upper']:.4f} < {POWER_THRESHOLD}.")
        print(f"  在 N={boot['n_valid']} 下能排除 |r|>={POWER_THRESHOLD} 的效应.")
        print("  零结果有力: 自然arm r≈0.02 从 'E 零结果' 升 'E + power 声明'.")
    else:
        verdict = "INSUFFICIENT_POWER"
        print(f"\n[verdict] |CI 上界|={boot['abs_ci_upper']:.4f} >= {POWER_THRESHOLD}.")
        print("  功效仍不足以排除 {POWER_THRESHOLD} 量级效应, 需更大 N 或报告降级.")

    out = {
        "experiment": "exp6_natural_arm_power",
        "model": MODEL, "k": K, "n_windows_large": N_WIN_LARGE,
        "n_tokens": raw["N"], "raw_file": RAW_NAME,
        "bootstrap": boot, "power_threshold": POWER_THRESHOLD,
        "verdict": verdict, "evidence_tier": "E",
        "note": ("phase6 自然arm扩样本 + cluster bootstrap by window 偏相关 CI. "
                 "复现 analyze_experimentC 偏相关口径 (控制 freq+norm+position). "
                 "CI 上界 = 能检出最小效应. 对应 limitation '自然arm E power pending'."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    outpath = OUT / "exp6_natural_arm_power.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\n[done] {outpath}  verdict={verdict}", flush=True)


if __name__ == "__main__":
    main()