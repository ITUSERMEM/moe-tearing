"""实验④: Qwen jump_rel 锚定重算 (shared-exclusive vs shared-inclusive).

purpose: 论文 limitation #3 后半 —— Qwen1.5-MoE 有 shared expert (始终激活, sigmoid 门控,
连续, 无 top-k 不连续), 但 paper repo 的 forward_hard 只合成 routed experts, 不含
shared. jump_rel = jump_abs / median||y||, 分母 ||y|| 是 block output norm. shared
对 jump_abs 贡献≈0 (shared 连续, 无 top-k 跳变), 但拉高 ||y|| 分母 -> shared-exclusive
口径的 jump_rel 系统性偏大. 本实验重算 shared-inclusive 版 jump_rel, 锚定该偏差量级.

design (复用 mtp 原语, 不改原仓库, 不调 run_sweep 避免重复loading模型):
  - loading Qwen bf16 一次 (device=cuda, 28.6GB, 32GB GPU 够)
  - discover_moe_blocks + capture_hidden_states_multi (SWEEP_PROBE_TEXTS)
  - 对 sweep_layers 每层, paired 对比两版:
      (a) exclusive: 原生 mtp.forward_hard (routed only, 论文 canonical 口径)
      (b) inclusive: monkey-patch mtp.forward_hard -> forward_hard_shared
                     (routed + sigmoid(shared_gate)*shared, Qwen 原生 block output 口径)
    两版用同 path_seed 的 find_boundary_pair random -> 同 h_a/h_b (paired)
  - continuity_signature 内 modes 元组调模块级 forward_hard, patch 后 hard_topk 模式
    自动用 shared 版 (soft_edge/tied 不受影响, 只对比 hard_topk.jump_rel)
  - jump_rel_excl vs jump_rel_incl: incl < excl (shared 拉高 ||y||, jump_abs unchanged)
  - 降幅 = (excl - incl) / excl, 锚定 limitation #3 后半偏差量级

算力: Qwen bf16 28.6GB, 32GB GPU 够. 每 layer blk.float() 仅浮点化当前层. m3_paths ×
n_layers × 2 版 决定耗时. m3_paths=50 × 2 layer × 2 = 200 continuity_signature (各含
3 模式 × 3 grid), ~15-30 min GPU.

env:
  MTEAR_MODEL (default /path/to/project/qwen)
  MTEAR_K (default 4)  M3_PATHS (default 50)  SWEEP_LAYERS (default "8,16")
  MTEAR_PROBE_DIR / MTEAR_OUT

产出: results/exp4_jump_rel_recalc.json, evidence_tier=E.
incl < excl -> shared 拉高 ||y|| 致 jump_rel 偏大, limitation #3 后半量化.
incl >= excl -> 反预期, 按反数据修补铁律报告降级.
"""
import sys, os, json
from pathlib import Path
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE_DIR = os.environ.get("MTEAR_PROBE_DIR", _HERE)
sys.path.insert(0, _PROBE_DIR)
import moe_tear_probe as mtp

MODEL = os.environ.get("MTEAR_MODEL", "/path/to/project/qwen")
K = int(os.environ.get("MTEAR_K", "4"))
M3_PATHS = int(os.environ.get("M3_PATHS", "50"))
SWEEP_LAYERS = os.environ.get("SWEEP_LAYERS", "8,16")
PATH_SEED = int(os.environ.get("MTEAR_PATH_SEED", "0"))
OUT = Path(os.environ.get("MTEAR_OUT", os.path.join(_HERE, "..", "results")))


def make_forward_hard_shared(orig_fh):
    """包装原生 forward_hard, 追加 Qwen shared expert (sigmoid 门控, 连续).
    OLMoE 无 shared_expert -> hasattr 守卫, 退化为原生 (与 exclusive 一致)."""
    def forward_hard_shared(block, h, k):
        out, mask = orig_fh(block, h, k)
        if hasattr(block, "shared_expert") and hasattr(block, "shared_expert_gate"):
            sg = torch.sigmoid(block.shared_expert_gate(h))
            out = out + sg * block.shared_expert(h)
        return out, mask
    return forward_hard_shared


def jump_rel_batch(blk, H, k, n_paths, path_seed, tau=0.15):
    """跑 n_paths 个 random boundary pair, 收集 hard_topk 模式的 jump_rel.
    paired: 同 path_seed -> 同一组 h_a/h_b (exclusive 与 inclusive 可比)."""
    gen = torch.Generator().manual_seed(path_seed)
    jump_rels = []
    for _ in range(n_paths):
        h_a, h_b, _ = mtp.find_boundary_pair(blk, H, k, mode="random",
                                             generator=gen, return_meta=True)
        sig = mtp.continuity_signature(blk, h_a, h_b, k=k, tau=tau)
        jump_rels.append(sig["hard_topk"]["jump_rel"])
    return np.array(jump_rels, dtype=float)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda"
    dtype = torch.bfloat16
    layers = [int(x) for x in SWEEP_LAYERS.split(",") if x.strip()]
    print(f"[④] jump_rel recalib: model={MODEL} k={K} m3_paths={M3_PATHS} "
          f"layers={layers} (Qwen bf16, GPU)", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    print("[④] loading Qwen bf16 到 GPU (28.6GB)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=dtype, device_map=device, low_cpu_mem_usage=True).eval()
    mtp.set_norm_topk(model, "auto")

    blocks = mtp.discover_moe_blocks(model)
    if not blocks:
        raise RuntimeError("未找到 MoE block")
    n_blocks = len(blocks)
    print(f"[④] {n_blocks} MoE blocks", flush=True)

    texts = mtp.SWEEP_PROBE_TEXTS
    target_blocks = {blocks[i][0]: blocks[i][1] for i in layers if 0 <= i < n_blocks}
    print(f"[④] capture hidden states (layers {layers})...", flush=True)
    states = mtp.capture_hidden_states_multi(model, tok, texts, target_blocks, device)

    orig_fh = mtp.forward_hard
    fh_shared = make_forward_hard_shared(orig_fh)

    per_layer = []
    for i in layers:
        if not (0 <= i < n_blocks):
            print(f"  [L{i}] 越界, skip", flush=True)
            continue
        name, blk, n_exp = blocks[i]
        H = states[name].float()
        try:
            blk.float()
            print(f"  [L{i} {name}] exclusive (routed only)...", flush=True)
            jr_excl = jump_rel_batch(blk, H, K, M3_PATHS, PATH_SEED)
            print(f"    jump_rel excl: median={np.median(jr_excl):.4f} "
                  f"IQR=[{np.percentile(jr_excl,25):.4f},{np.percentile(jr_excl,75):.4f}]",
                  flush=True)

            mtp.forward_hard = fh_shared
            try:
                print(f"  [L{i} {name}] inclusive (routed + shared)...", flush=True)
                jr_incl = jump_rel_batch(blk, H, K, M3_PATHS, PATH_SEED)
            finally:
                mtp.forward_hard = orig_fh
            print(f"    jump_rel incl: median={np.median(jr_incl):.4f} "
                  f"IQR=[{np.percentile(jr_incl,25):.4f},{np.percentile(jr_incl,75):.4f}]",
                  flush=True)

            med_excl = float(np.median(jr_excl))
            med_incl = float(np.median(jr_incl))
            drop = (med_excl - med_incl) / max(abs(med_excl), 1e-9)
            per_layer.append({
                "layer_idx": i, "block_name": name, "n_experts": n_exp,
                "n_paths": M3_PATHS,
                "jump_rel_exclusive": {"median": med_excl,
                                       "iqr": [float(np.percentile(jr_excl, 25)),
                                               float(np.percentile(jr_excl, 75))],
                                       "mean": float(np.mean(jr_excl))},
                "jump_rel_inclusive": {"median": med_incl,
                                       "iqr": [float(np.percentile(jr_incl, 25)),
                                               float(np.percentile(jr_incl, 75))],
                                       "mean": float(np.mean(jr_incl))},
                "drop_frac": drop,
                "per_path_excl": jr_excl.tolist(),
                "per_path_incl": jr_incl.tolist(),
            })
            print(f"  [L{i}] drop_frac = {drop:.3f}  "
                  f"(incl {med_incl:.4f} vs excl {med_excl:.4f})", flush=True)
        except Exception as e:
            print(f"  [L{i}] FAILED: {e}", flush=True)
            per_layer.append({"layer_idx": i, "block_name": name, "error": str(e)})
        finally:
            blk.to(dtype)

    drops = [p["drop_frac"] for p in per_layer if "drop_frac" in p]
    mean_drop = float(np.mean(drops)) if drops else float("nan")
    print("\n" + "=" * 70)
    print(f"jump_rel 降幅 (shared 拉高 ||y||): mean drop_frac = {mean_drop:.3f}")
    for p in per_layer:
        if "drop_frac" in p:
            print(f"  L{p['layer_idx']}: excl={p['jump_rel_exclusive']['median']:.4f} "
                  f"incl={p['jump_rel_inclusive']['median']:.4f} "
                  f"drop={p['drop_frac']:.3f}")
    print("=" * 70)

    if drops and mean_drop > 0:
        verdict = "SHARED_INFLATES_DENOMINATOR"
        print(f"\n[verdict] inclusive jump_rel < exclusive (drop={mean_drop:.3f}).")
        print("  shared 拉高 ||y|| 分母, 论文 shared-exclusive jump_rel 系统性偏大.")
        print("  limitation #3 后半量化: jump_rel 锚定偏差 = drop_frac.")
    elif drops and mean_drop <= 0:
        verdict = "REVERSED"
        print(f"\n[verdict] inclusive >= exclusive (drop={mean_drop:.3f})! 反预期.")
        print("  按反数据修补铁律报告, 不重写叙事. shared 对 jump_rel 的影响需降级审查.")
    else:
        verdict = "NO_VALID_LAYERS"

    out = {
        "experiment": "exp4_jump_rel_recalc",
        "model": MODEL, "k": K, "dtype": "bf16", "m3_paths": M3_PATHS,
        "path_seed": PATH_SEED, "sweep_layers": layers,
        "per_layer": per_layer, "mean_drop_frac": mean_drop,
        "verdict": verdict, "evidence_tier": "E",
        "note": ("Qwen shared-exclusive (forward_hard routed only, 论文 canonical) "
                 "vs shared-inclusive (routed + sigmoid(shared_gate)*shared, Qwen 原生 "
                 "block output 口径). jump_rel=jump_abs/median||y||, shared 连续拉高分母. "
                 "paired: 同 path_seed boundary pair. 对应 limitation #3 后半."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    outpath = OUT / "exp4_jump_rel_recalc.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\n[done] {outpath}  verdict={verdict}", flush=True)


if __name__ == "__main__":
    main()