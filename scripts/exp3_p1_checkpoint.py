"""实验③: P1 checkpoint-trajectory adjudication (acquired vs intrinsic).

purpose: 论文 registered prediction P1 —— 35× gap 主因定位在源头expertoutput范数差 (466×),
但该范数差是训练习得 (acquired) 还是架构固有 (intrinsic) 未定. 判据:
  - 若 acquired: 早期 OLMoE checkpoint 的 Δflip 应显著不同于最终 checkpoint
  - 若 intrinsic: Δflip 跨 checkpoint 平坦
OLMoE checkpoint 已发布, 单 GPU 推理即可 (论文原话 "single GPU suffices"). 做了把
P1 从 tier P 升 C/E.

design (复用 phase3 三控制, env 驱动多 checkpoint):
  - 外层循环 checkpoint (env MTEAR_CKPTS 逗号分隔的 HF id 或本地路径)
  - 每 checkpoint: monkey-patch p3.MODEL + p3.load_model, 跑 p3.main() 得 experimentA
    算 Δflip@L8 = median(cross[8] - near[8]) (源头跳变, energy 口径)
  - 聚合: Δflip vs checkpoint-step 轨迹, flat -> intrinsic, changing -> acquired
  - 注: checkpoint-step 需用户在 MTEAR_CKPT_STEPS 提供对应训练steps数 (同序)

算力: OLMoE-1B-7B bf16 14GB, 32GB GPU 够. 每 checkpoint phase3 跑 N_TOKENS 窗 × 4
forward. checkpoint 数 × N_TOKENS 决定总耗时. 建议 3-5 个 checkpoint × N_TOKENS=300.

env:
  MTEAR_CKPTS (必填, 逗号分隔 checkpoint id/路径, 如 ckpt_1000,ckpt_50000,allenai/OLMoE-1B-7B-0924)
  MTEAR_CKPT_STEPS (逗号分隔对应训练steps数, 如 1000,50000,final)
  N_TOKENS (default 300)  MTEAR_K (default 8)
  MTEAR_PROBE_DIR / MTEAR_SCRIPTS_DIR / MTEAR_OUT

产出: results/exp3_p1_checkpoint.json {per_ckpt: {step, dflip_L8}}, evidence_tier.
flat -> intrinsic (P1 falsified acquired); changing -> acquired (P1 confirmed).
"""
import sys, os, json, shutil, tempfile, subprocess
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import snapshot_download

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE_DIR = os.environ.get("MTEAR_PROBE_DIR", _HERE)
sys.path.insert(0, _PROBE_DIR)
import moe_tear_probe as mtp
sys.path.insert(0, os.environ.get("MTEAR_SCRIPTS_DIR", _HERE))
import phase3_experimentA as p3

CKPTS = os.environ.get("MTEAR_CKPTS", "").split(",")
CKPT_STEPS = os.environ.get("MTEAR_CKPT_STEPS", "").split(",")
K = int(os.environ.get("MTEAR_K", "8"))
N_TOKENS = int(os.environ.get("N_TOKENS", "300"))
OUT = Path(os.environ.get("MTEAR_OUT", os.path.join(_HERE, "..", "results")))


def dflip_from_experimentA(a_json_path):
    """Δflip@L8 = median(cross[8] - near[8]) over valid rows (源头跳变 energy 口径)."""
    A = json.loads(Path(a_json_path).read_text())
    rows = [r for r in A["rows"] if r.get("ok", True)]
    dflips = [r["cross"]["energy"][8] - r["near"]["energy"][8] for r in rows]
    ddirs = [r["near"]["energy"][8] - r["orth"]["energy"][8] for r in rows]
    return {"dflip_L8": float(np.median(dflips)),
            "ddir_L8": float(np.median(ddirs)),
            "n_valid": len(rows),
            "dflip_iqr": [float(np.percentile(dflips, 25)), float(np.percentile(dflips, 75))]}


def resolve_ckpt(ckpt_spec, idx):
    # 支持 "repo@revision" 语法: OLMoE-0924 中间 checkpoint 是同 repo 的 branch, 非独立 repo.
    # 无 @ 则视为普通 repo id 或本地路径, 原行为unchanged.
    if "@" not in ckpt_spec:
        return ckpt_spec, None
    repo, rev = ckpt_spec.split("@", 1)
    cache_dir = OUT / f"_ckpt_cache_{idx}"
    print(f"  [dl] snapshot_download {repo}@{rev} -> {cache_dir}", flush=True)
    local = snapshot_download(
        repo_id=repo, revision=rev,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*"],
        cache_dir=str(cache_dir))
    return local, cache_dir


def run_one_checkpoint(ckpt_id, step_label, idx):
    """loading checkpoint, 跑 phase3 三控制, 返 Δflip@L8. 跑完即删临时 cache 释放磁盘."""
    os.environ["DTYPE"] = "bf16"
    os.environ["MTEAR_K"] = str(K)
    local_model, temp_cache = resolve_ckpt(ckpt_id, idx)
    os.environ["MTEAR_MODEL"] = local_model
    p3.MODEL = local_model
    p3.K = K
    p3.DTYPE = "bf16"
    p3.SUFFIX = f"_ckpt{idx}"
    p3.OUT = OUT
    p3.N_TOKENS = N_TOKENS
    a_name = f"experimentA_ckpt{idx}_{step_label}.json"
    p3.A_OUTNAME = a_name
    print(f"\n[③] checkpoint {idx+1}/{len(CKPTS)}: {ckpt_id} (step={step_label})", flush=True)
    p3.main()
    # 释放 GPU
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # 跑完即删临时 checkpoint cache (每 ckpt 13.8GB), 小 JSON 留 OUT 不删, 磁盘峰值 ~14GB
    if temp_cache is not None and Path(temp_cache).exists():
        shutil.rmtree(temp_cache, ignore_errors=True)
        print(f"  [cleanup] 删除临时 cache {temp_cache}", flush=True)
    return dflip_from_experimentA(OUT / a_name), a_name


def main():
    ckpts = [c.strip() for c in CKPTS if c.strip()]
    steps = [s.strip() for s in CKPT_STEPS if s.strip()] or [f"ckpt{i}" for i in range(len(ckpts))]
    # 子进程模式: --single IDX 只跑第 IDX 个 checkpoint 写 JSON 退出 (彻底释放显存)
    if len(sys.argv) >= 3 and sys.argv[1] == "--single":
        idx = int(sys.argv[2])
        run_one_checkpoint(ckpts[idx], steps[idx], idx)
        return
    if len(ckpts) < 2:
        print("[③] 需至少 2 个 checkpoint (MTEAR_CKPTS 逗号分隔). 至少一个早期 + 最终.", flush=True)
        print("    例: MTEAR_CKPTS=allenai/OLMoE-1B-7B-0924@step5000-tokens20B,allenai/OLMoE-1B-7B-0924@main "
              "MTEAR_CKPT_STEPS=5000,final")
        return
    print(f"[③] P1 checkpoint adjudication: {len(ckpts)} ckpts, K={K}, N_TOKENS={N_TOKENS}", flush=True)
    print(f"    ckpts={ckpts}", flush=True)
    print(f"    steps={steps}", flush=True)

    per_ckpt = []
    for i, (ckpt, step) in enumerate(zip(ckpts, steps)):
        a_name = f"experimentA_ckpt{i}_{step}.json"
        # 续传: 已有 JSON skip (ckpt0 已跑完则不重跑)
        if (OUT / a_name).exists():
            print(f"\n[续传] {a_name} 已存在, skip子进程", flush=True)
        else:
            print(f"\n[③] 启动子进程跑 checkpoint {i+1}/{len(ckpts)}: {ckpt} (step={step})", flush=True)
            ret = subprocess.run([sys.executable, os.path.abspath(__file__), "--single", str(i)])
            if ret.returncode != 0:
                print(f"  [错误] ckpt{i} 子进程失败 rc={ret.returncode}", flush=True)
                continue
        stats = dflip_from_experimentA(OUT / a_name)
        per_ckpt.append({"ckpt": ckpt, "step": step, **stats, "a_file": a_name})
        print(f"  -> Δflip@L8={stats['dflip_L8']:+.4f}  Δdir@L8={stats['ddir_L8']:+.4f}  "
              f"n={stats['n_valid']}", flush=True)

    # 轨迹verdict
    dflips = [p["dflip_L8"] for p in per_ckpt]
    dflip_range = max(dflips) - min(dflips)
    dflip_median = float(np.median(dflips))
    # 相对变化: range / median (排除最终点作基准更稳, 这里用全局 median)
    rel_change = dflip_range / max(abs(dflip_median), 1e-9)

    print("\n" + "=" * 70)
    print("Δflip@L8 轨迹:")
    for p in per_ckpt:
        print(f"  step={p['step']:<12} Δflip={p['dflip_L8']:+.4f}  (IQR {p['dflip_iqr'][0]:+.4f},{p['dflip_iqr'][1]:+.4f})")
    print(f"\n  range={dflip_range:.4f}  rel_change={rel_change:.2f}×", flush=True)

    if rel_change < 0.2:
        verdict = "INTRINSIC"
        print(f"\n[verdict] Δflip 跨 checkpoint 平坦 (rel_change={rel_change:.2f}<0.2) -> 范数差架构固有.")
        print("  P1 falsified acquired: 35× gap 主因是 intrinsic, 非训练习得. P1 tier P->E (falsified).")
    else:
        verdict = "ACQUIRED"
        print(f"\n[verdict] Δflip 跨 checkpoint 显著变化 (rel_change={rel_change:.2f}>=0.2) -> 范数差训练习得.")
        print("  P1 confirmed acquired: 35× gap 主因是 training-acquired. P1 tier P->C/E.")

    out = {
        "experiment": "exp3_p1_checkpoint",
        "k": K, "n_tokens": N_TOKENS, "n_ckpts": len(per_ckpt),
        "per_ckpt": per_ckpt,
        "trajectory": {"dflip_range": dflip_range, "rel_change": rel_change,
                       "dflip_median": dflip_median},
        "verdict": verdict, "evidence_tier": "E",
        "note": ("P1 checkpoint-trajectory adjudication. Δflip@L8=median(cross-near) energy. "
                 "flat->intrinsic (acquired falsified); changing->acquired (confirmed). "
                 "需 OLMoE 早期 checkpoint (MTEAR_CKPTS). 论文 P1 'single GPU suffices'."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    outpath = OUT / "exp3_p1_checkpoint.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\n[done] {outpath}  verdict={verdict}", flush=True)


if __name__ == "__main__":
    main()