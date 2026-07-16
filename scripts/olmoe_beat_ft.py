"""olmoe_beat_ft.py — OLMoE BEAT 微调 (单层 L8, boundaryalignment)

design:
  - loading OLMoE-1B-7B-0125
  - 冻结全部parameter, 仅解冻 L8 层expert FFN (gate_up_proj + down_proj)
  - BEAT regularization: ‖e_i - e_j‖²/(‖e_i‖²+‖e_j‖²) 仅boundary token
  - controlarm: λ=0 (仅 CE), λ=0.1, λ=1.0

output: results/olmoe_beat_ft_l{LAMBDA}.json
环境: OLLM_LAMBDA, OLLM_STEPS, OLLM_LR, OLLM_BS
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_ID = "allenai/OLMoE-1B-7B-0125"

LAMBDA = float(os.environ.get("OLLM_LAMBDA", "0.1"))
STEPS = int(os.environ.get("OLLM_STEPS", "500"))
LR = float(os.environ.get("OLLM_LR", "1e-4"))
BS = int(os.environ.get("OLLM_BS", "2"))
WIN = 256
STEP = 128
TAU = 0.02
TARGET_LAYER = 8

OUT_TAG = os.environ.get("OLLM_TAG", f"l{str(LAMBDA).replace('.','_')}")
OUT_PATH = os.path.join(OUT_DIR, f"olmoe_beat_ft_{OUT_TAG}.json")
VAL_OFFSET = 600000

# === Data ===
DATA_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "datasets--wikitext", "snapshots", "b08601e04326c79dfdd32d625aee71d232d685c3",
    "wikitext-103-raw-v1")

def load_wikitext103_tokens(offset=0, max_tokens=None):
    import pandas as pd
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
    if offset > 0:
        ids = ids[offset:]
    if max_tokens is not None:
        ids = ids[:max_tokens]
    return ids

def discover_moe_blocks(model):
    """返回 [(name, module, num_experts)] for all MoE blocks"""
    return [(name, mod, mod.experts.num_experts) for name, mod in model.named_modules() if isinstance(mod, OlmoeSparseMoeBlock)]

def expert_forward_olmoe(blk, e, h):
    gate, up = F.linear(h.float(), blk.experts.gate_up_proj[e].float()).chunk(2, dim=-1)
    act = F.silu(gate) * up
    return F.linear(act, blk.experts.down_proj[e].float())

# === BEAT alignmentregularization (仅目标层) ===
def compute_align_loss_single(blk, tau, k, device):
    if not hasattr(blk, "_last_logits") or not hasattr(blk, "_last_captured_h"):
        return torch.zeros((), device=device)
    logits = blk._last_logits.detach()
    zs = logits.sort(dim=-1, descending=True)
    d_t = zs.values[:, k-1] - zs.values[:, k]
    mask = (d_t < tau).cpu()
    if not mask.any():
        return torch.zeros((), device=device)
    ek = zs.indices[:, k-1][mask]
    ek1 = zs.indices[:, k][mask]
    h_b = blk._last_captured_h[mask].to(device)
    total, count = 0.0, 0
    for idx in range(mask.sum().item()):
        e_i, e_j = int(ek[idx]), int(ek1[idx])
        yi = expert_forward_olmoe(blk, e_i, h_b[idx:idx+1])
        yj = expert_forward_olmoe(blk, e_j, h_b[idx:idx+1])
        yi = yi.reshape(-1); yj = yj.reshape(-1)
        num = (yi - yj).norm()**2
        den = yi.norm()**2 + yj.norm()**2 + 1e-8
        total += num / den
        count += 1
    return (total / max(count, 1)).to(device) if count > 0 else torch.zeros((), device=device)

# === 增强measurement (复用 granite_pams_align 的 measure_all 逻辑, 适配 OLMoE) ===
def measure_all(model, ids, moe_blocks, k, n_win=20, device="cuda"):
    model.eval()
    n_layers = len(moe_blocks)
    all_h = {i: [] for i in range(n_layers)}
    all_ek = {i: [] for i in range(n_layers)}
    all_ek1 = {i: [] for i in range(n_layers)}
    all_dt = {i: [] for i in range(n_layers)}
    with torch.no_grad():
        for wi in range(n_win):
            wid = ids[wi*STEP:wi*STEP+WIN].unsqueeze(0).to(device)
            model(**dict(input_ids=wid, attention_mask=torch.ones_like(wid)))
            for i, (_, blk, _) in enumerate(moe_blocks):
                if hasattr(blk, "_last_captured_h") and hasattr(blk, "_last_captured_logits"):
                    all_h[i].append(blk._last_captured_h)
                    logits = blk._last_captured_logits
                    zs = logits.sort(dim=-1, descending=True)
                    all_ek[i].append(zs.indices[:, k-1])
                    all_ek1[i].append(zs.indices[:, k])
                    d_t = zs.values[:, k-1] - zs.values[:, k]
                    all_dt[i].append(d_t)
    results = []
    for i, (_, blk, _) in enumerate(moe_blocks):
        h = torch.cat(all_h[i]).to(device)
        ek = torch.cat(all_ek[i])
        ek1 = torch.cat(all_ek1[i])
        d_t_all = torch.cat(all_dt[i]).numpy()
        yk, yk1 = torch.zeros_like(h), torch.zeros_like(h)
        for e_idx in range(blk.experts.num_experts):
            m_k = (ek == e_idx); m_k1 = (ek1 == e_idx)
            if m_k.any(): yk[m_k] = expert_forward_olmoe(blk, e_idx, h[m_k])
            if m_k1.any(): yk1[m_k1] = expert_forward_olmoe(blk, e_idx, h[m_k1])
        num = (yk - yk1).norm(dim=-1); den = yk.norm(dim=-1) + yk1.norm(dim=-1) + 1e-6
        tear = (num / den).detach().cpu().numpy()
        cos_raw = ((yk * yk1).sum(dim=-1) / (yk.norm(dim=-1) * yk1.norm(dim=-1) + 1e-6)).detach().cpu().numpy()
        n_boundary = int((d_t_all < 0.05).sum())
        T = len(h)
        # 全局 pairwise cos
        n_anchors = min(32, len(h))
        anchor_idx = torch.randperm(len(h))[:n_anchors].to(device)
        all_e_out = []
        for e_idx in range(blk.experts.num_experts):
            out = expert_forward_olmoe(blk, e_idx, h[anchor_idx])
            all_e_out.append(out.detach().cpu())
        all_e_out = torch.stack(all_e_out).reshape(blk.experts.num_experts, -1)
        cos_matrix = torch.mm(F.normalize(all_e_out, dim=-1), F.normalize(all_e_out, dim=-1).T)
        tri = torch.triu(cos_matrix, diagonal=1)
        nonzero = tri[tri > -2]
        avg_pairwise_cos = float(nonzero.mean()) if nonzero.numel() > 0 else 0.0
        results.append({
            "layer": i, "n_tokens": T,
            "tear_median": float(np.nanmedian(tear)),
            "cos_median": float(np.nanmedian(cos_raw)),
            "boundary_frac_lt0.05": float(n_boundary / T) if T > 0 else 0.0,
            "pairwise_cos_avg": avg_pairwise_cos,
        })
    model.train()
    return results

def measure_val_loss(model, ids, n_win=20, device="cuda"):
    model.eval()
    losses = []
    with torch.no_grad():
        for wi in range(0, n_win, BS):
            n = min(BS, n_win - wi)
            wids = torch.stack([ids[(wi+j)*STEP:(wi+j)*STEP+WIN].to(device) for j in range(n)])
            out = model(**dict(input_ids=wids, attention_mask=torch.ones_like(wids)))
            l = F.cross_entropy(out.logits[:,:-1].float().reshape(-1, out.logits.size(-1)), wids[:,1:].reshape(-1), reduction="none")
            losses.append(l.mean().item())
    model.train()
    return float(np.mean(losses))

def main():
    torch.manual_seed(42); np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[olmoe_beat_ft] λ={LAMBDA} steps={STEPS} bs={BS} lr={LR} target_layer={TARGET_LAYER}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[init] loading OLMoE-1B-7B-0125...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=device, torch_dtype=torch.bfloat16)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    moe_blocks = discover_moe_blocks(model)
    k = 8
    print(f"[init] {len(moe_blocks)} MoE blocks, k={k}", flush=True)

    # === 冻结全部, 仅解冻 L8 expert FFN ===
    for p in model.parameters():
        p.requires_grad = False
    target_blk = moe_blocks[TARGET_LAYER][1]
    target_blk.experts.gate_up_proj.requires_grad = True
    target_blk.experts.down_proj.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[init] trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M total ({trainable/total*100:.1f}%)", flush=True)

    # === Hook: 抓 h + logits (全部 MoE 层均需抓, 但仅目标层 BEAT 参与梯度) ===
    def make_hook(blk):
        def hook(mod, inp, out):
            h = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            blk._last_captured_h = h.cpu()
            router_logits = out[0].detach().float()  # OLMoE gate 返回 (logits, weights, indices), fp32
            blk._last_captured_logits = router_logits.cpu()
            blk._last_logits = router_logits
        return hook
    handles = [blk.gate.register_forward_hook(make_hook(blk)) for _, blk, _ in moe_blocks]

    ids = load_wikitext103_tokens()
    ids_val = ids[VAL_OFFSET:VAL_OFFSET+WIN*50]

    # === measurement前 ===
    print(f"\n=== BEFORE ===", flush=True)
    before_all = measure_all(model, ids_val, moe_blocks, k, n_win=20, device=device)
    before_val = measure_val_loss(model, ids_val, n_win=20, device=device)
    M2_before = np.mean([l["tear_median"] for l in before_all if not np.isnan(l["tear_median"])])
    print(f"[before] val_loss={before_val:.4f} M2_avg={M2_before:.4f}", flush=True)
    for r in before_all[:5]:
        print(f"  L{r['layer']}: tear={r['tear_median']:.4f} bnd={r['boundary_frac_lt0.05']:.3f}", flush=True)

    # === 训练 ===
    n_train_ids = min(len(ids), 200000)
    train_starts = list(range(0, (n_train_ids-WIN)//STEP*STEP, STEP))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)
    loss_log, val_losses = [], []
    best_val_loss = float('inf')
    t0 = time.time()

    for step in range(STEPS):
        start = train_starts[step % len(train_starts)]
        wid = ids[start:start+WIN].unsqueeze(0).to(device)
        out = model(**dict(input_ids=wid, attention_mask=torch.ones_like(wid)))
        ce = F.cross_entropy(out.logits[0, :-1].float(), wid[0, 1:])

        # BEAT loss (仅目标层)
        align = compute_align_loss_single(target_blk, TAU, k, device)
        total = ce + LAMBDA * align
        if torch.isnan(total) or torch.isinf(total):
            print(f"[step {step}] NaN! ce={ce:.4f} align={align:.4f}", flush=True); break
        total.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step(); optimizer.zero_grad(); scheduler.step()
        loss_log.append(float(ce.detach()))

        if (step+1) % 100 == 0:
            ck_val = measure_val_loss(model, ids_val, n_win=10, device=device)
            val_losses.append({"step": step+1, "val_loss": ck_val})
            if ck_val < best_val_loss:
                best_val_loss = ck_val
            print(f"[{step+1}/{STEPS}] ce={np.mean(loss_log[-50:]):.4f} align={float(align.detach()):.4f} "
                  f"val_loss={ck_val:.4f} mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB "
                  f"{time.time()-t0:.0f}s", flush=True)

    # === measurement后 ===
    print(f"\n=== AFTER ===", flush=True)
    after_all = measure_all(model, ids_val, moe_blocks, k, n_win=20, device=device)
    after_val = measure_val_loss(model, ids_val, n_win=20, device=device)
    M2_after = np.mean([l["tear_median"] for l in after_all if not np.isnan(l["tear_median"])])
    M2_reduction = (M2_before - M2_after) / M2_before if M2_before > 0 else 0

    print(f"  M2: {M2_before:.4f} → {M2_after:.4f} ({M2_reduction*100:+.1f}%)", flush=True)
    print(f"  val_loss: {before_val:.4f} → {after_val:.4f}", flush=True)

    result = {"model": MODEL_ID, "lambda": LAMBDA, "steps": STEPS, "target_layer": TARGET_LAYER,
        "M2_before": M2_before, "M2_after": M2_after, "M2_reduction": M2_reduction,
        "before_val_loss": before_val, "after_val_loss": after_val, "best_val_loss": best_val_loss,
        "trainable_params_M": trainable/1e6,
        "before": {"full": before_all},
        "after": {"full": after_all},
        "val_loss_trace": val_losses}
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] {OUT_PATH}", flush=True)
    for h in handles: h.remove()

if __name__ == "__main__":
    main()
