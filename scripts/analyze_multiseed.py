"""analyze_multiseed.py — 多 seed 结果分析 + cross-seed Jaccard 零线

用法: python3 scripts/analyze_multiseed.py
需: 多 seed 6 arm全部done (weights + JSON)
"""

import os, json, torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeMoE, GraniteMoeForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "..", "weights")
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-base"
DEVICE = "cuda"

def load_val_data():
    DATA_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
        "datasets--wikitext", "snapshots", "b08601e04326c79dfdd32d625aee71d232d685c3",
        "wikitext-103-raw-v1")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    texts = []
    for fn in ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]:
        df = pd.read_parquet(os.path.join(DATA_CACHE, fn))
        texts.extend(t for t in df["text"] if isinstance(t, str) and len(t.strip()) > 0)
    all_ids = []
    for i in range(0, len(texts), 1000):
        for ids in tok(texts[i:i+1000], add_special_tokens=False)["input_ids"]:
            all_ids.extend(ids)
    ids = torch.tensor(all_ids, dtype=torch.long)
    ids = ids[600000:600000+128*20]
    return ids.to(DEVICE)

def get_routing(model, ids):
    moe = [(n,m) for n,m in model.named_modules() if isinstance(m, GraniteMoeMoE)]
    routing = {}
    def make_hook(li):
        def hook(mod, inp, out):
            routing[li] = out[-1].detach().topk(8, dim=-1).indices.cpu()
        return hook
    handles = [m.router.register_forward_hook(make_hook(i)) for i, (_, m) in enumerate(moe)]
    model.eval()
    with torch.no_grad():
        for wi in range(10):
            w = ids[wi*64:wi*64+128].unsqueeze(0)
            model(**dict(input_ids=w, attention_mask=torch.ones_like(w)))
    for h in handles: h.remove()
    return routing

def compute_expert_swap(model, ids):
    from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeMoE
    moe = [(n,m) for n,m in model.named_modules() if isinstance(m, GraniteMoeMoE)]
    def efg(blk, e, h):
        wg = blk.input_linear.weight[e].float(); wd = blk.output_linear.weight[e].float()
        g, u = torch.nn.functional.linear(h, wg).chunk(2, dim=-1)
        return torch.nn.functional.linear(blk.activation(g) * u, wd)
    def make_h(blk):
        def hook(mod, inp, out):
            blk._h = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
        return hook
    handles = [m.router.register_forward_hook(make_h(m)) for _, m in moe]
    k = 8
    all_jumps = []
    with torch.no_grad():
        for wi in range(10):
            w = ids[wi*64:wi*64+128].unsqueeze(0)
            model(**dict(input_ids=w, attention_mask=torch.ones_like(w)))
            for _, blk in moe:
                h = blk._h.to(DEVICE)
                out = blk.router(h)
                logits = out[0]
                zs = logits.sort(dim=-1, descending=True)
                dt = zs.values[:, k-1] - zs.values[:, k]
                boundary = (dt < 0.02).cpu()
                if not boundary.any(): continue
                for idx in torch.where(boundary)[0]:
                    hi = h[idx:idx+1]
                    yn = sum(efg(blk, int(zs.indices[idx, r]), hi) for r in range(k))
                    yr = sum(efg(blk, int(zs.indices[idx, r]), hi) for r in range(k-1)) + efg(blk, int(zs.indices[idx, k]), hi)
                    all_jumps.append(((yn - yr).norm() / yn.norm().clamp(min=1e-8)).item())
    for h in handles: h.remove()
    return float(np.median(all_jumps)) if all_jumps else None

def main():
    # Load all JSONs
    arms = {
        "l0_0_s42": {"lam": 0.0, "seed": 42, "steps": 20000},
        "l0_0": {"lam": 0.0, "seed": 42, "steps": 20000},  # legacy tag
    }
    for lam, base in [(0.0, "l0"), (1.0, "l10")]:
        for s in [43, 44, 45]:
            arms[f"{base}_s{s}"] = {"lam": lam, "seed": s, "steps": 10000}

    ids = None
    results = {"val_ce": {}, "m2": {}}
    jaccard_pairs = []

    for tag, info in arms.items():
        json_path = os.path.join(OUT_DIR, f"granite_pams_align_{tag}.json")
        wt_path = os.path.join(WEIGHTS_DIR, f"granite_pams_align_{tag}")
        if not os.path.exists(json_path):
            # try legacy name (l0_0 maps to l0_0 without seed)
            if tag == "l0_0":
                continue  # already handled above
            print(f"[skip] {tag}: JSON not found", flush=True)
            continue
        if not os.path.exists(wt_path):
            if tag == "l0_0":
                continue
            print(f"[skip] {tag}: weights not found", flush=True)
            continue

        with open(json_path) as f:
            d = json.load(f)
        av = d["after"]["val_loss"]
        bv = d["before"]["val_loss"]
        m2_avg = np.mean([l["tear_median"] for l in d["after"]["full"] if not np.isnan(l["tear_median"])])
        m2_before = np.mean([l["tear_median"] for l in d["before"]["full"] if not np.isnan(l["tear_median"])])
        results["val_ce"][tag] = {"before": bv, "after": av, "best": d.get("best_val_loss", av)}
        results["m2"][tag] = {"before": m2_before, "after": m2_avg, "reduction": (m2_before - m2_avg) / m2_before * 100}

        # Cross-seed Jaccard: compare λ=0 models against each other
        if info["lam"] == 0.0 and info["steps"] == 10000:
            if ids is None:
                ids = load_val_data()
            print(f"[jaccard] loading {tag}...", flush=True)
            model = GraniteMoeForCausalLM.from_pretrained(wt_path, device_map=DEVICE, torch_dtype=torch.bfloat16)
            routing = get_routing(model, ids)
            jaccard_pairs.append((tag, info["seed"], routing, model))
            del model; torch.cuda.empty_cache()

    # Compute cross-seed Jaccard
    print(f"\n=== Cross-seed Jaccard (λ=0, seed pairs at 10k steps) ===", flush=True)
    jaccard_results = {}
    for i in range(len(jaccard_pairs)):
        for j in range(i+1, len(jaccard_pairs)):
            t1, s1, r1, _ = jaccard_pairs[i]
            t2, s2, r2, _ = jaccard_pairs[j]
            key = f"seed{s1}_vs_seed{s2}"
            js = []
            for li in range(24):
                a = r1[li].numpy(); b = r2[li].numpy()
                js.append(float(np.mean([
                    len(set(a[t]) & set(b[t])) / len(set(a[t]) | set(b[t]))
                    for t in range(len(a))])))
            avg = float(np.mean(js))
            jaccard_results[key] = {"avg": avg, "per_layer_range": [min(js), max(js)]}
            print(f"  {key}: {avg:.4f} [{min(js):.4f}, {max(js):.4f}]", flush=True)

    # Print summary table
    print(f"\n=== Multi-seed summary ===", flush=True)
    for tag in sorted(results["val_ce"].keys()):
        v = results["val_ce"][tag]
        m = results["m2"].get(tag, {})
        lam = arms.get(tag, {}).get("lam", "?")
        seed = arms.get(tag, {}).get("seed", "?")
        print(f"  λ={lam} seed={seed}: val_CE {v['before']:.4f}→{v['after']:.4f} (best={v['best']:.4f}) "
              f"M2 {m.get('before',0):.4f}→{m.get('after',0):.4f} ({m.get('reduction',0):+.1f}%)", flush=True)

    # Save
    out = {"val_ce": results["val_ce"], "m2": results["m2"], "jaccard_cross_seed": jaccard_results}
    out_path = os.path.join(OUT_DIR, "multiseed_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] {out_path}", flush=True)

if __name__ == "__main__":
    main()
