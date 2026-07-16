"""实验⑤: shared dose-response N 扩样本 (176 -> 700-1000).

purpose: 论文 limitation #7 —— phase4c shared dose-response 只用了 176 valid rows
(A_FILE 的有效样本), 功效不足, mock review 质询 "dose-response 曲线在更大 N 下是否
稳定, Wilcoxon p 是否仍显著". 本实验把 N 扩到 ~700-1000, 重跑 dose-response, verification
alpha 单调性 + per_L8 vs global 分离在扩样本下成立.

design (复用 phase3 + phase4c, env 驱动大 N, 不改原仓库):
  - Step 1: 跑 phase3 用 N_TOKENS_LARGE (default 5000) 产出 experimentA_<N>_qwen.json
            (valid 率 ~17.6%, 5000 token -> ~880 valid rows)
  - Step 2: patch phase4c.A_FILE 指向大 N 的 A 文件, 跑 phase4c.main()
            -> dose-response N~880 (global + per_L8 × 4 alpha = 8 run_b)
  - Step 3: 读原 phase4c result (N=176) 对比 dose-response 曲线:
            - alpha 单调性是否稳定
            - Wilcoxon p 是否更显著 (N 大 -> p 更小)
            - per_L8 vs global 分离是否仍成立

算力: Qwen bf16 28.6GB, 32GB GPU 够. Step1 phase3 5000 token × 3 控制 ~37 min;
Step2 8 run_b × ~880 valid × 4 forward ~1.5h; 总 ~2h. 一次一实验, 跑前confirm无其他
GPU 占用. N_TOKENS_LARGE env 可先调 2000 verification再扩 5000.

env:
  MTEAR_MODEL (default /path/to/project/qwen)  MTEAR_K (default 4)
  N_TOKENS_LARGE (default 5000)
  MTEAR_PROBE_DIR / MTEAR_SCRIPTS_DIR / MTEAR_OUT
  MTEAR_ORIG_4C (default experimentB_shared_ablation_qwen.json, 原 N=176 基准)

产出: results/exp5_dose_response_N.json + experimentA_<N>_qwen.json +
      experimentB_shared_ablation_qwen_N.json, evidence_tier=E.
曲线稳定 + p 更显著 -> limitation #7 功效升级; 曲线反转 -> 铁律报告降级.
"""
import sys, os, json
from pathlib import Path
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE_DIR = os.environ.get("MTEAR_PROBE_DIR", _HERE)
sys.path.insert(0, _PROBE_DIR)
import moe_tear_probe as mtp
sys.path.insert(0, os.environ.get("MTEAR_SCRIPTS_DIR", _HERE))
import phase3_experimentA as p3
import phase4c_qwen_shared_ablation as p4c

MODEL = os.environ.get("MTEAR_MODEL", "/path/to/project/qwen")
K = int(os.environ.get("MTEAR_K", "4"))
N_LARGE = int(os.environ.get("N_TOKENS_LARGE", "5000"))
OUT = Path(os.environ.get("MTEAR_OUT", os.path.join(_HERE, "..", "results")))
ORIG_4C = os.environ.get("MTEAR_ORIG_4C", "experimentB_shared_ablation_qwen.json")


def run_phase3_large_n():
    """跑 phase3 大 N_TOKENS, 产出大 N 的 A 文件, 返 (a_name, n_valid)."""
    a_name = f"experimentA_{N_LARGE}_qwen.json"
    os.environ["DTYPE"] = "bf16"
    os.environ["MTEAR_MODEL"] = MODEL
    os.environ["MTEAR_K"] = str(K)
    p3.MODEL = MODEL
    p3.K = K
    p3.DTYPE = "bf16"
    p3.N_TOKENS = N_LARGE
    p3.SUFFIX = f"_{N_LARGE}"
    p3.A_OUTNAME = a_name
    p3.OUT = OUT
    print(f"[⑤] Step1: phase3 N_TOKENS={N_LARGE} -> {a_name}", flush=True)
    p3.main()
    A = json.loads((OUT / a_name).read_text())
    n_valid = sum(1 for r in A["rows"] if r.get("ok", True))
    print(f"[⑤] Step1 done: {n_valid} valid rows (A_FILE={a_name})", flush=True)
    return a_name, n_valid


def run_phase4c_large_n(a_name):
    """patch phase4c 指向大 N A 文件, 跑 dose-response, 返 out_file 名."""
    out_file = f"experimentB_shared_ablation_qwen_N{N_LARGE}.json"
    p4c.MODEL = MODEL
    p4c.A_FILE = a_name
    p4c.OUT = OUT
    p4c.OUT_FILE = out_file
    # L/K/DELTA 沿用 phase3 模块级默认 (Qwen L=8 K=4), 保持与原 phase4c 一致
    print(f"[⑤] Step2: phase4c dose-response (A_FILE={a_name}, OUT={out_file})", flush=True)
    p4c.main()
    return out_file


def compare_to_orig(out_file):
    """读大 N dose-response + 原 N=176 基准, 对比曲线稳定性 + p 显著性."""
    new = json.loads((OUT / out_file).read_text())
    orig_path = OUT / ORIG_4C
    orig = json.loads(orig_path.read_text()) if orig_path.exists() else None
    alphas = new["alphas"]

    def extract(res):
        out = {}
        for mode in ["global", "per_L8"]:
            out[mode] = {a: res["results"][mode][str(a)]["summary"] for a in alphas}
        return out

    n_new = extract(new)
    n_orig = extract(orig) if orig else None

    print("\n" + "=" * 80)
    print("dose-response 对比 (大 N vs 原 N=176):")
    for mode in ["global", "per_L8"]:
        print(f"\n  [{mode}]")
        print(f"    {'alpha':>6} {'N':>5} {'Δflip_med(new)':>16} {'p(new)':>10} "
              f"{'Δflip_med(orig)':>16} {'p(orig)':>10}")
        for a in alphas:
            s_new = n_new[mode][a]
            row = f"    {a:>6.2f} {s_new['N']:>5} {s_new['dflip_median']:>16.6f} {s_new['dflip_wilcoxon_p']:>10.3e}"
            if n_orig:
                s_o = n_orig[mode][a]
                row += f" {s_o['dflip_median']:>16.6f} {s_o['dflip_wilcoxon_p']:>10.3e}"
            print(row)
    print("=" * 80)

    # 单调性: per_L8 Δflip 应随 alpha 单调 (a=0 最大, a=1 最小)
    def monotonic(d):
        vals = [d[a]["dflip_median"] for a in alphas]
        return all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    mono_new = monotonic(n_new["per_L8"])
    mono_orig = monotonic(n_orig["per_L8"]) if n_orig else None
    # per_L8 vs global 分离: a=0 时 per_L8 Δflip > global (注入层隔离更强)
    sep_new = n_new["per_L8"][0.0]["dflip_median"] > n_new["global"][0.0]["dflip_median"]
    # p 显著性: N 大 -> p 更小 (至少不更大)
    p_new = n_new["per_L8"][0.0]["dflip_wilcoxon_p"]
    p_orig = n_orig["per_L8"][0.0]["dflip_wilcoxon_p"] if n_orig else None
    p_improved = (p_orig is None) or (p_new <= p_orig)

    return {
        "n_valid_new": n_new["per_L8"][alphas[0]]["N"],
        "n_valid_orig": n_orig["per_L8"][alphas[0]]["N"] if n_orig else None,
        "monotonic_per_L8_new": mono_new,
        "monotonic_per_L8_orig": mono_orig,
        "separation_per_L8_gt_global_new": sep_new,
        "p_per_L8_a0_new": p_new,
        "p_per_L8_a0_orig": p_orig,
        "p_improved_with_N": p_improved,
        "ratio_a0_a1_new": n_new["per_L8"][0.0]["dflip_median"] /
                           max(n_new["per_L8"][1.0]["dflip_median"], 1e-12),
        "ratio_a0_a1_orig": (n_orig["per_L8"][0.0]["dflip_median"] /
                             max(n_orig["per_L8"][1.0]["dflip_median"], 1e-12)) if n_orig else None,
    }


def main():
    print(f"[⑤] dose-response N 扩样本: model={MODEL} k={K} N_TOKENS_LARGE={N_LARGE}", flush=True)
    a_name, n_valid = run_phase3_large_n()
    if n_valid < 200:
        print(f"[⑤] 警告: valid rows={n_valid} 偏少, 建议 N_TOKENS_LARGE 调更大.", flush=True)
    out_file = run_phase4c_large_n(a_name)
    cmp = compare_to_orig(out_file)

    print(f"\n[对比汇总] N: {cmp['n_valid_orig']} -> {cmp['n_valid_new']}")
    print(f"  per_L8 单调性: orig={cmp['monotonic_per_L8_orig']} new={cmp['monotonic_per_L8_new']}")
    print(f"  per_L8>global 分离 (new): {cmp['separation_per_L8_gt_global_new']}")
    print(f"  p(a=0): orig={cmp['p_per_L8_a0_orig']} new={cmp['p_per_L8_a0_new']} improved={cmp['p_improved_with_N']}")
    print(f"  ratio a0/a1: orig={cmp['ratio_a0_a1_orig']} new={cmp['ratio_a0_a1_new']}")

    if cmp["monotonic_per_L8_new"] and cmp["p_improved_with_N"]:
        verdict = "STABLE_AT_LARGE_N"
        print(f"\n[verdict] 大 N 下 dose-response 单调性稳定 + p 改善 -> limitation #7 功效升级.")
    elif cmp["monotonic_per_L8_new"]:
        verdict = "MONOTONIC_BUT_P_NOT_IMPROVED"
        print(f"\n[verdict] 单调性稳定但 p 未改善, 部分升级 limitation #7.")
    else:
        verdict = "MONOTONICITY_BROKEN"
        print(f"\n[verdict] 大 N 下单调性破坏! 按反数据修补铁律报告, dose-response 主张需降级.")

    out = {
        "experiment": "exp5_dose_response_N",
        "model": MODEL, "k": K, "n_tokens_large": N_LARGE,
        "a_file": a_name, "ablation_file": out_file,
        "orig_4c_baseline": ORIG_4C,
        "comparison": cmp, "verdict": verdict, "evidence_tier": "E",
        "note": ("phase4c shared dose-response N 扩样本. Step1 phase3 大 N_TOKENS 产出 "
                 "valid rows, Step2 phase4c 复用. 对比 N=176 基准曲线稳定性 + Wilcoxon p. "
                 "对应 limitation #7 功效不足."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    outpath = OUT / "exp5_dose_response_N.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\n[done] {outpath}  verdict={verdict}", flush=True)


if __name__ == "__main__":
    main()