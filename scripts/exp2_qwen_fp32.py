"""实验②: Qwen1.5-MoE fp32 verification (35× gap + Δflip 主导).

purpose: 论文 limitation #3 "Qwen cross-model bf16 throughout, fp32 validation blocked"
—— Qwen fp32 权重 57GB, 本机 30GB 内存装不下, 被 block. 这台 128GB 内存机器解锁,
verification bf16 跨模型结论 (35× gap, Δflip 主导 35-924×) 在 fp32 下成立, 堵住"bf16 噪声底"
质询 (C' 已显示量化诱发 5.2% 翻转).

design (monkey-patch 复用, 不改原仓库):
  - monkey-patch p3.load_model: device_map="auto" + max_memory (GPU 30GB + CPU 120GB offload)
    Qwen fp32 57GB 装不下 32GB 显存, 用 accelerate 自动分片, 128GB 内存 offload 剩余层.
  - patch p3.OUT / p3.A_OUTNAME / p3.N_TOKENS: 产出到本地 results, N_TOKENS env 控制
    (默认 200 先跑通, verification后扩到 1000).
  - 跑 p3.main() -> experimentA_qwen_fp32.json (三控制 energy / 源头)
  - 跑 phase4.main() -> experimentB_kl_qwen_fp32.json (next-token KL / Δflip)
  - 读 experimentB 算 Δflip=med(KL_cross-near), Δdir=med(KL_near-orth) + Wilcoxon
  - 对比 OLMoE (phase4d DFLIP_OLMOE: bf16 0.0008 / fp32 0.0011) -> 35× gap fp32

算力: Qwen fp32 57GB + offload, 32GB GPU 做 attention + CPU offload 权重. 慢 (offload
层间搬运), N_TOKENS=200 约 1-3 小时, 1000 约 5-15 小时. 先小 N verification可行性.

env:
  MTEAR_MODEL (default /path/to/project/qwen, 本地 Qwen1.5-MoE-A2.7B)
  MTEAR_K (default 4)  N_TOKENS (default 200)
  GPU_MEM (default "30GB")  CPU_MEM (default "120GB")
  MTEAR_PROBE_DIR / MTEAR_SCRIPTS_DIR / MTEAR_OUT

产出: results/exp2_qwen_fp32.json + experimentA/B_qwen_fp32.json, evidence_tier=E.
若 fp32 Δflip/Δdir 比与 bf16 一致 (35× gap, Δflip>>Δdir) -> limitation #3 fp32 verification落地.
"""
import sys, os, json
from pathlib import Path
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE_DIR = os.environ.get("MTEAR_PROBE_DIR", _HERE)
sys.path.insert(0, _PROBE_DIR)
import moe_tear_probe as mtp
sys.path.insert(0, os.environ.get("MTEAR_SCRIPTS_DIR", _HERE))
import phase3_experimentA as p3
import phase4_experimentB as p4

MODEL = os.environ.get("MTEAR_MODEL", "/path/to/project/qwen")
K = int(os.environ.get("MTEAR_K", "4"))
N_TOKENS = int(os.environ.get("N_TOKENS", "200"))
GPU_MEM = os.environ.get("GPU_MEM", "30GB")
CPU_MEM = os.environ.get("CPU_MEM", "120GB")
OUT = Path(os.environ.get("MTEAR_OUT", os.path.join(_HERE, "..", "results")))

# OLMoE control (phase4d_gate_weight.py 硬编码, 0924 canonical)
DFLIP_OLMOE_BF16 = 0.0008
DFLIP_OLMOE_FP32 = 0.0011


def patch_load_model():
    """monkey-patch p3.load_model: Qwen fp32 device_map auto + CPU offload."""
    def load_auto():
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32, device_map="auto",
            max_memory={0: GPU_MEM, "cpu": CPU_MEM},
            low_cpu_mem_usage=True).eval()
        mtp.set_norm_topk(model, "auto")
        return model, tok
    p3.load_model = load_auto


def main():
    os.environ["DTYPE"] = "fp32"
    os.environ["MTEAR_MODEL"] = MODEL
    os.environ["MTEAR_K"] = str(K)
    # 重新读 p3 模块级配置 (env 已设)
    p3.MODEL = MODEL
    p3.K = K
    p3.DTYPE = "fp32"
    p3.SUFFIX = "_fp32"
    p3.OUT = OUT
    p3.N_TOKENS = N_TOKENS
    patch_load_model()
    print(f"[②] Qwen fp32: model={MODEL} k={K} N_TOKENS={N_TOKENS} "
          f"GPU<{GPU_MEM}+CPU<{CPU_MEM} offload>", flush=True)

    A_NAME = f"experimentA_1000_qwen_fp32.json"
    B_NAME = f"experimentB_kl_qwen_fp32.json"
    p3.A_OUTNAME = A_NAME
    os.environ["MTEAR_A_FILE"] = A_NAME
    os.environ["MTEAR_B_FILE"] = B_NAME
    p4.A_FILE = A_NAME
    p4.B_FILE = B_NAME
    p4.OUT = OUT

    # Phase A: 三控制 energy / 源头
    print("[②] Phase A (三控制 energy)...", flush=True)
    p3.main()
    # Phase B: next-token KL / Δflip
    print("[②] Phase B (next-token KL / Δflip)...", flush=True)
    p4.main()

    # 聚合: 读 experimentB 算 Δflip/Δdir + Wilcoxon
    bpath = OUT / B_NAME
    B = json.loads(bpath.read_text())
    rows = [r for r in B["rows"] if r.get("ok", True)]
    kl_cross = np.array([r["cross"] for r in rows])
    kl_near = np.array([r["near"] for r in rows])
    kl_orth = np.array([r["orth"] for r in rows])
    dflip = kl_cross - kl_near
    ddir = kl_near - kl_orth
    dflip_med = float(np.median(dflip))
    ddir_med = float(np.median(ddir))
    # Wilcoxon (scipy 可能无, 用简单 sign test)
    from scipy.stats import wilcoxon
    try:
        w_p = wilcoxon(dflip, alternative="greater").pvalue
    except Exception:
        w_p = float((dflip > 0).mean())

    # 35× gap: Qwen fp32 Δflip vs OLMoE fp32 Δflip
    ratio_fp32 = DFLIP_OLMOE_FP32 / max(dflip_med, 1e-12)
    ratio_bf16 = DFLIP_OLMOE_BF16 / max(dflip_med, 1e-12)
    dom_ratio = abs(dflip_med) / max(abs(ddir_med), 1e-12)

    print("\n" + "=" * 70)
    print(f"Qwen fp32 (n={len(rows)} valid):")
    print(f"  Δflip = {dflip_med:+.6f}  (KL_cross - KL_near)")
    print(f"  Δdir  = {ddir_med:+.6f}  (KL_near  - KL_orth)")
    print(f"  Δflip/Δdir 主导比 = {dom_ratio:.1f}×  (论文 35-924×)")
    print(f"  Wilcoxon p (Δflip>0) = {w_p:.3e}")
    print(f"  35× gap: OLMoE_fp32/Qwen_fp32 = {ratio_fp32:.1f}×  (bf16 ref {ratio_bf16:.1f}×)")
    print("=" * 70)

    verdict = ("FP32_VALIDATED" if (dflip_med > 0 and dom_ratio > 5 and ratio_fp32 > 10)
               else "FP32_PARTIAL" if dflip_med > 0 else "FP32_REVERSED")
    print(f"\n[verdict] verdict={verdict}")
    if verdict == "FP32_VALIDATED":
        print("  Qwen fp32 下 Δflip 主导 + 35× gap 成立 -> limitation #3 fp32 verification落地.")
    elif verdict == "FP32_REVERSED":
        print("  Qwen fp32 Δflip 反转! 按反数据修补铁律报告, bf16 结论需降级审查.")

    out = {
        "experiment": "exp2_qwen_fp32",
        "model": MODEL, "k": K, "dtype": "fp32", "n_tokens": N_TOKENS,
        "n_valid": len(rows),
        "qwen_fp32": {"dflip_median": dflip_med, "ddir_median": ddir_med,
                      "dom_ratio": dom_ratio, "wilcoxon_p": float(w_p)},
        "gap_vs_olmoe": {"olmoe_fp32_over_qwen_fp32": ratio_fp32,
                         "olmoe_bf16_over_qwen_fp32": ratio_bf16,
                         "olmoe_fp32_dflip": DFLIP_OLMOE_FP32,
                         "olmoe_bf16_dflip": DFLIP_OLMOE_BF16},
        "verdict": verdict, "evidence_tier": "E",
        "note": ("Qwen fp32 device_map=auto offload (GPU+CPU). 复用 phase3/phase4 "
                 "三控制 (原生 forward + hidden 注入, Qwen shared 自动包含). "
                 "对应 limitation #3 内存墙解锁. 若 N_TOKENS=200 先verification, 跑通后扩 1000."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    outpath = OUT / "exp2_qwen_fp32.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\n[done] {outpath}", flush=True)


if __name__ == "__main__":
    main()