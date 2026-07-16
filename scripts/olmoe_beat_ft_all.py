"""olmoe_beat_ft_all.py — OLMoE BEAT 全层微调

loading OLMoE-1B-7B-0125, 解冻所有 16 层expert FFN,
BEAT regularization操作在所有层上, Adafactor 优化器.

output: results/olmoe_beat_ft_all_l{LAMBDA}.json
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
LR = float(os.environ.get("OLLM_LR", "5e-5"))
BS = int(os.environ.get("OLLM_BS", "2"))
WIN = 256
STEP = 128
TAU = 0.02

OUT_TAG = os.environ.get("OLLM_TAG", f"all_l{str(LAMBDA).replace('.','_')}")
OUT_PATH = os.path.join(OUT_DIR, f"olmoe_beat_ft_all_{OUT_TAG}.json")
VAL_OFFSET = 600000
TRAIN_HOLDOUT_OFFSET = 500000

DATA_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "datasets--wikitext", "snapshots", "b08601e04326c79dfdd32d625aee71d232d685c3",
    "wikitext-103-raw-v1")

def load_tokens(offset=0, max_tokens=None):
    import pandas as pd
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = []
    for fn in ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]:
        df = pd.read_parquet(os.path.join(DATA_CACHE, fn))
        texts.extend(t for t in df["text"] if isinstance(t, str) and len(t.strip()) > 0)
    all_ids = []
    for i in range(0, len(texts), 1000):
        for ids in tokenizer(texts[i:i+1000], add_special_tokens=False)["input_ids"]:
            all_ids.extend(ids)
    ids = torch.tensor(all_ids, dtype=torch.long)
    if offset > 0: ids = ids[offset:]
    if max_tokens is not None: ids = ids[:max_tokens]
    return ids

def discover_moe_blocks(model):
    return [(name, mod, mod.experts.num_experts) for name, mod in model.named_modules() if isinstance(mod, OlmoeSparseMoeBlock)]

def expert_forward(blk, e, h):
    g, u = F.linear(h.float(), blk.experts.gate_up_proj[e].float()).chunk(2, dim=-1)
    return F.linear(F.silu(g) * u, blk.experts.down_proj[e].float())

def compute_align_loss(blocks, tau, k, device):
    total_loss, total_cnt = 0.0, 0
    for _, blk, _ in blocks:
        if not hasattr(blk, "_last_captured_h"):
            continue
        h = blk._last_captured_h.to(device)  # [T, D] fp32, detached
        w_gate = blk.gate.weight.detach().float()  # [E, D] fp32, 停梯度
        logits = F.linear(h, w_gate)  # [T, E] fp32
        zs = logits.sort(dim=-1, descending=True)
        d_t = zs.values[:, k-1] - zs.values[:, k]
        mask = (d_t < tau).cpu()
        if not mask.any(): continue
        ek = zs.indices[:, k-1][mask]
        ek1 = zs.indices[:, k][mask]
        h_b = blk._last_captured_h[mask].to(device)
        for idx in range(mask.sum().item()):
            yi = expert_forward(blk, int(ek[idx]), h_b[idx:idx+1])
            yj = expert_forward(blk, int(ek1[idx]), h_b[idx:idx+1])
            n = (yi - yj).norm() ** 2
            d = yi.norm() ** 2 + yj.norm() ** 2 + 1e-8
            total_loss += n / d; total_cnt += 1
    return (total_loss / max(total_cnt, 1)).to(device) if total_cnt > 0 else torch.zeros((), device=device)

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
                    all_dt[i].append(zs.values[:, k-1] - zs.values[:, k])
    results = []
    for i, (_, blk, _) in enumerate(moe_blocks):
        h = torch.cat(all_h[i]).to(device)
        ek = torch.cat(all_ek[i]); ek1 = torch.cat(all_ek1[i])
        d_t_all = torch.cat(all_dt[i]).numpy()
        yk, yk1 = torch.zeros_like(h), torch.zeros_like(h)
        for e_idx in range(blk.experts.num_experts):
            m_k = (ek == e_idx); m_k1 = (ek1 == e_idx)
            if m_k.any(): yk[m_k] = expert_forward(blk, e_idx, h[m_k])
            if m_k1.any(): yk1[m_k1] = expert_forward(blk, e_idx, h[m_k1])
        num = (yk - yk1).norm(dim=-1); den = yk.norm(dim=-1) + yk1.norm(dim=-1) + 1e-6
        tear = (num / den).detach().cpu().numpy()
        n_boundary = int((d_t_all < 0.05).sum())
        results.append({
            "layer": i, "n_tokens": len(h),
            "tear_median": float(np.nanmedian(tear)),
            "boundary_frac_lt0.05": float(n_boundary / len(h)) if len(h) > 0 else 0.0,
        })
    model.train()
    return results

def measure_loss(model, ids, n_win=20, device="cuda"):
    model.eval(); losses = []
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
    print(f"[olmoe_beat_ft_all] λ={LAMBDA} steps={STEPS} bs={BS} lr={LR}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print(f"[init] loading OLMoE-1B-7B-0125...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=device, torch_dtype=torch.bfloat16)
    model.train()
    model.gradient_checkpointing_enable()

    moe_blocks = discover_moe_blocks(model)
    k = 8
    print(f"[init] {len(moe_blocks)} MoE blocks", flush=True)

    # 冻结全部, 解冻所有层expert FFN
    for p in model.parameters():
        p.requires_grad = False
    for _, blk, _ in moe_blocks:
        blk.experts.gate_up_proj.requires_grad = True
        blk.experts.down_proj.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[init] trainable: {trainable/1e6:.0f}M / {sum(p.numel() for p in model.parameters())/1e6:.0f}M total", flush=True)

    # Hook: 所有层
    def make_hook(blk):
        def hook(mod, inp, out):
            blk._last_captured_h = inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu()
            blk._last_captured_logits = out[0].detach().float().cpu()
        return hook
    handles = [blk.gate.register_forward_hook(make_hook(blk)) for _, blk, _ in moe_blocks]

    ids = load_tokens()
    ids_train = load_tokens(offset=TRAIN_HOLDOUT_OFFSET, max_tokens=WIN*20)
    ids_val = load_tokens(offset=VAL_OFFSET, max_tokens=WIN*20)

    # measurement前
    print(f"\n=== BEFORE ===", flush=True)
    before_all = measure_all(model, ids_val, moe_blocks, k, n_win=5, device=device)
    before_train = measure_loss(model, ids_train, n_win=5, device=device)
    before_val = measure_loss(model, ids_val, n_win=5, device=device)
    M2_before = np.mean([l["tear_median"] for l in before_all if not np.isnan(l["tear_median"])])
    print(f"[before] train={before_train:.4f} val={before_val:.4f} M2={M2_before:.4f}", flush=True)

    # 训练
    n_train_ids = min(len(ids), 200000)
    train_starts = list(range(0, (n_train_ids-WIN)//STEP*STEP, STEP))
    optimizer = torch.optim.Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)
    loss_log = []; best_val = float('inf')
    t0 = time.time()

    for step in range(STEPS):
        start = train_starts[step % len(train_starts)]
        wid = ids[start:start+WIN].unsqueeze(0).to(device)
        out = model(**dict(input_ids=wid, attention_mask=torch.ones_like(wid)))
        ce = F.cross_entropy(out.logits[0, :-1].float(), wid[0, 1:])
        align = compute_align_loss(moe_blocks, TAU, k, device)
        total = ce + LAMBDA * align
        if torch.isnan(total) or torch.isinf(total):
            print(f"[step {step}] NaN! ce={ce:.4f}", flush=True); break
        total.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step(); optimizer.zero_grad(); scheduler.step()
        loss_log.append(float(ce.detach()))
        if (step+1) % 100 == 0:
            print(f"[{step+1}/{STEPS}] ce={np.mean(loss_log[-50:]):.4f} align={float(align.detach()):.4f} "
                  f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB {time.time()-t0:.0f}s", flush=True)

    # measurement后
    print(f"\n=== AFTER ===", flush=True)
    after_all = measure_all(model, ids_val, moe_blocks, k, n_win=5, device=device)
    after_train = measure_loss(model, ids_train, n_win=5, device=device)
    after_val = measure_loss(model, ids_val, n_win=5, device=device)
    M2_after = np.mean([l["tear_median"] for l in after_all if not np.isnan(l["tear_median"])])
    M2_reduction = (M2_before - M2_after) / M2_before if M2_before > 0 else 0

    print(f"  M2: {M2_before:.4f} -> {M2_after:.4f} ({M2_reduction*100:+.1f}%)", flush=True)
    print(f"  train: {before_train:.4f} -> {after_train:.4f}", flush=True)
    print(f"  val:   {before_val:.4f} -> {after_val:.4f}", flush=True)

    result = {"model": MODEL_ID, "lambda": LAMBDA, "steps": STEPS,
        "M2_before": M2_before, "M2_after": M2_after, "M2_reduction": M2_reduction,
        "before": {"full": before_all, "loss": before_train, "val_loss": before_val},
        "after": {"full": after_all, "loss": after_train, "val_loss": after_val}}
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] {OUT_PATH}", flush=True)
    for h in handles: h.remove()

if __name__ == "__main__":
    main()
