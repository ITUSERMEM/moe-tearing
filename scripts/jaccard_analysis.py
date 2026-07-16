"""jaccard_analysis.py — 跨armrouting一致率 (基于已存最终权重)

measurement: 对每层 top-8 routing决策, 计算 λ=0 / 0.1 / 1.0 两两之间 Jaccard 相似度
"""

import os, json, torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, GraniteMoeForCausalLM
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeMoE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "..", "weights")
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-base"
PAMS_WIN = 128
PAMS_STEP = 64
VAL_OFFSET = 600000

def load_val_tokens(device):
    import pandas as pd
    DATA_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
        "datasets--wikitext", "snapshots", "b08601e04326c79dfdd32d625aee71d232d685c3",
        "wikitext-103-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = []
    for fn in ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]:
        fp = os.path.join(DATA_CACHE, fn)
        df = pd.read_parquet(fp)
        texts.extend(t for t in df["text"] if isinstance(t, str) and len(t.strip()) > 0)
    all_ids = []
    for i in range(0, len(texts), 1000):
        enc = tokenizer(texts[i:i+1000], add_special_tokens=False)["input_ids"]
        for ids in enc: all_ids.extend(ids)
    ids = torch.tensor(all_ids, dtype=torch.long)
    ids = ids[VAL_OFFSET:VAL_OFFSET+PAMS_WIN*20]
    return ids.to(device)

def capture_routing(model, ids, k=8):
    """返回 {layer_i: [token_idx, k] 的 top-k expert索引}"""
    moe_blocks = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, GraniteMoeMoE)]
    routing = {}
    hooks = []
    for layer_idx, (_, blk) in enumerate(moe_blocks):
        def make_hook(li):
            def hook(mod, inp, out):
                logits = out[-1].detach()
                z = logits.sort(dim=-1, descending=True)
                topk = z.indices[:, :k].cpu()
                if not hasattr(model, "_routing_cache"):
                    model._routing_cache = {}
                model._routing_cache[li] = topk
            return hook
        hooks.append(blk.router.register_forward_hook(make_hook(layer_idx)))
    model._routing_cache = {}
    with torch.no_grad():
        n_win = len(ids) // PAMS_STEP
        for wi in range(n_win):
            wid = ids[wi*PAMS_STEP:wi*PAMS_STEP+PAMS_WIN].unsqueeze(0)
            model(**dict(input_ids=wid, attention_mask=torch.ones_like(wid)))
    routing = dict(model._routing_cache)
    for h in hooks: h.remove()
    return routing

def jaccard_similarity(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[jaccard] device={device}", flush=True)
    ids = load_val_tokens(device)
    print(f"[jaccard] {len(ids)} validation tokens", flush=True)

    arms = {"l0_0": 0.0, "l0_1": 0.1, "l1_0": 1.0}
    all_routing = {}

    for tag, lam in arms.items():
        wt_path = os.path.join(WEIGHTS_DIR, f"granite_pams_align_{tag}")
        print(f"[jaccard] loading {tag} (λ={lam}) from {wt_path}", flush=True)
        model = GraniteMoeForCausalLM.from_pretrained(wt_path, device_map=device, torch_dtype=torch.bfloat16)
        model.eval()
        routing = capture_routing(model, ids, k=8)
        all_routing[tag] = routing
        del model; torch.cuda.empty_cache()
        print(f"[jaccard] {tag}: {len(routing)} layers captured", flush=True)

    pairs = [("l0_0","l0_1"), ("l0_0","l1_0"), ("l0_1","l1_0")]
    n_layers = len(all_routing["l0_0"])
    config = AutoConfig.from_pretrained(MODEL_ID)

    results = {"n_tokens": len(ids), "k": 8}
    for a, b in pairs:
        pair_jaccards = []
        pair_overlap_rates = []
        for li in range(n_layers):
            ra = all_routing[a][li].numpy()  # [T, k]
            rb = all_routing[b][li].numpy()
            T = ra.shape[0]
            jaccards = []
            overlaps = []
            for t in range(T):
                sa = set(ra[t])
                sb = set(rb[t])
                jaccards.append(jaccard_similarity(sa, sb))
                overlaps.append(len(sa & sb) / config.num_experts_per_tok)
            pair_jaccards.append({
                "layer": li,
                "jaccard_mean": float(np.mean(jaccards)),
                "jaccard_median": float(np.median(jaccards)),
                "jaccard_p5": float(np.percentile(jaccards, 5)),
                "jaccard_p95": float(np.percentile(jaccards, 95)),
                "overlap_rate": float(np.mean(overlaps)),
            })
        avg_j = float(np.mean([p["jaccard_mean"] for p in pair_jaccards]))
        print(f"[jaccard] {a} vs {b}: Jaccard={avg_j:.4f} (per-layer range: {min(p[\"jaccard_mean\"] for p in pair_jaccards):.4f}-{max(p[\"jaccard_mean\"] for p in pair_jaccards):.4f})", flush=True)
        results[f"{a}_vs_{b}"] = {"per_layer": pair_jaccards, "avg_jaccard": avg_j}

    out_path = os.path.join(OUT_DIR, "jaccard_cross_arm.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[jaccard] done → {out_path}", flush=True)

if __name__ == "__main__":
    main()
