"""expert_swap_demo.py — expert替换jump幅度演示

loading Granite 三arm终态权重, 对比 λ=0 vs λ=1.0 的替换后jump幅度.
jump = ‖f_normal - f_replaced‖ / ‖f_normal‖
"""

import os, json, torch, torch.nn.functional as F
import numpy as np, pandas as pd
from transformers import AutoConfig, AutoTokenizer
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeMoE, GraniteMoeForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "..", "weights")
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-base"
DEVICE = "cuda"
WIN = 128; STEP = 64; N_WIN = 10
TAU = 0.02

def load_val_tokens():
    DATA_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
        "datasets--wikitext", "snapshots", "b08601e04326c79dfdd32d625aee71d232d685c3",
        "wikitext-103-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    texts = []
    for fn in ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]:
        df = pd.read_parquet(os.path.join(DATA_CACHE, fn))
        texts.extend(t for t in df["text"] if isinstance(t, str) and len(t.strip()) > 0)
    all_ids = []
    for i in range(0, len(texts), 1000):
        for ids in tokenizer(texts[i:i+1000], add_special_tokens=False)["input_ids"]:
            all_ids.extend(ids)
    ids = torch.tensor(all_ids, dtype=torch.long)
    ids = ids[600000:600000+WIN*N_WIN]
    return ids

def expert_forward_granite(blk, e, h):
    w_gu = blk.input_linear.weight[e].float()
    w_dn = blk.output_linear.weight[e].float()
    g, u = F.linear(h, w_gu).chunk(2, dim=-1)
    return F.linear(blk.activation(g) * u, w_dn)

def measure_swap_jump(model, ids, device=DEVICE):
    """返回每层的替换jump幅度列表"""
    moe_blocks = [(n,m) for n,m in model.named_modules() if isinstance(m, GraniteMoeMoE)]
    k = model.config.num_experts_per_tok
    all_jumps = {i: [] for i in range(len(moe_blocks))}

    # Hook to capture h + logits
    def make_hook(blk):
        def hook(mod, inp, out):
            blk._hook_h = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            blk._hook_logits = out[-1].detach().float()
        return hook
    handles = [b.router.register_forward_hook(make_hook(b)) for _, b in moe_blocks]

    model.eval()
    with torch.no_grad():
        for wi in range(N_WIN):
            wid = ids[wi*STEP:wi*STEP+WIN].unsqueeze(0).to(device)
            model(**dict(input_ids=wid, attention_mask=torch.ones_like(wid)))

            for li, (_, blk) in enumerate(moe_blocks):
                h = blk._hook_h.to(device)
                logits = blk._hook_logits.to(device)
                zs = logits.sort(dim=-1, descending=True)
                d_t = zs.values[:, k-1] - zs.values[:, k]
                boundary = (d_t < TAU).cpu()
                if not boundary.any(): continue

                for idx in torch.where(boundary)[0]:
                    hi = h[idx:idx+1]
                    # 正常 top-k output
                    y_normal = torch.zeros(hi.shape[0], hi.shape[1], device=device)
                    for rank in range(k):
                        e = int(zs.indices[idx, rank])
                        y_normal += expert_forward_granite(blk, e, hi)

                    # 替换: 用第 k+1 名替换第 k 名
                    e_k = int(zs.indices[idx, k-1])
                    e_k1 = int(zs.indices[idx, k])
                    y_replace = torch.zeros_like(y_normal)
                    for rank in range(k-1):
                        e = int(zs.indices[idx, rank])
                        y_replace += expert_forward_granite(blk, e, hi)
                    # 用 e_{k+1} 替代 e_k
                    y_replace += expert_forward_granite(blk, e_k1, hi)

                    jump = (y_normal - y_replace).norm() / y_normal.norm().clamp(min=1e-8)
                    all_jumps[li].append(float(jump.item()))

    for h in handles: h.remove()
    return all_jumps

def main():
    for tag, lam in [("l0_0", 0.0), ("l1_0", 1.0)]:
        wt_path = os.path.join(WEIGHTS_DIR, f"granite_pams_align_{tag}")
        print(f"\n[swap] loading {tag} (λ={lam}) from {wt_path}", flush=True)
        model = GraniteMoeForCausalLM.from_pretrained(wt_path, device_map=DEVICE, torch_dtype=torch.bfloat16)
        ids = load_val_tokens().to(DEVICE)
        jumps = measure_swap_jump(model, ids)

        per_layer = {}
        all_jumps_flat = []
        for li, jlist in jumps.items():
            per_layer[li] = {
                "n": len(jlist),
                "jump_median": float(np.median(jlist)) if jlist else None,
                "jump_mean": float(np.mean(jlist)) if jlist else None,
                "jump_p90": float(np.percentile(jlist, 90)) if jlist else None,
            }
            all_jumps_flat.extend(jlist)
            if len(jlist) > 0:
                print(f"  L{li}: n={len(jlist)} median={np.median(jlist):.4f}", flush=True)

        result = {"model_tag": tag, "lambda": lam, "n_tokens": len(ids),
            "global_median": float(np.median(all_jumps_flat)) if all_jumps_flat else None,
            "global_mean": float(np.mean(all_jumps_flat)) if all_jumps_flat else None,
            "per_layer": per_layer}
        out_path = os.path.join(OUT_DIR, f"expert_swap_{tag}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[swap] done -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
