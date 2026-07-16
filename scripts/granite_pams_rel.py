"""granite_pams_rel.py — 改进格A: 相对margin PAMS + B 口径主终点

改动 vs granite_pams_from_scratch.py:
  1. regularization: relu(ε_rel - d_t/σ) 代替 relu(ε - d_t), σ=|z[k-1]|+|z[k]|
  2. 主终点: B 口径 (固定 h, 仅 gate 精度变)
  3. 副终点: A 口径 + d_t_rel 分布

环境: PAMS_LAMBDA, PAMS_STEPS, PAMS_LR, PAMS_MODE=rel|abs (def rel)
output: results/granite_pams_rel_l{LAMBDA}.json
"""
import os, sys, json, gc, time, math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeMoE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-base"

PAMS_LAMBDA = float(os.environ.get("PAMS_LAMBDA", "0"))
PAMS_STEPS = int(os.environ.get("PAMS_STEPS", "5000"))
PAMS_EPS_REL = float(os.environ.get("PAMS_EPS_REL", "0.01"))
PAMS_LR = float(os.environ.get("PAMS_LR", "3e-4"))
PAMS_BS = int(os.environ.get("PAMS_BS", "4"))
PAMS_WIN = 128
PAMS_STEP = 64

OUT_TAG = os.environ.get("PAMS_TAG", f"l{str(PAMS_LAMBDA).replace('.','_')}")
OUT_PATH = os.path.join(OUT_DIR, f"granite_pams_rel_{OUT_TAG}.json")

# === Data ===
DATA_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "datasets--wikitext", "snapshots", "b08601e04326c79dfdd32d625aee71d232d685c3",
    "wikitext-103-raw-v1")

def load_wikitext103(tokenizer, max_examples=None):
    import pandas as pd
    texts = []
    for fn in ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]:
        fp = os.path.join(DATA_CACHE, fn)
        df = pd.read_parquet(fp)
        for text in df["text"]:
            if isinstance(text, str) and len(text.strip()) > 0:
                texts.append(text)
                if max_examples and len(texts) >= max_examples: break
        if max_examples and len(texts) >= max_examples: break
    all_ids = []
    for i in range(0, len(texts), 1000):
        batch = texts[i:i+1000]
        enc = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for ids in enc: all_ids.extend(ids)
    ids = torch.tensor(all_ids, dtype=torch.long)
    print(f"[data] {len(ids)} tokens ({len(texts)} lines)", flush=True)
    return ids

def discover_moe_blocks(model):
    return [(name, mod, mod.router.num_experts) for name, mod in model.named_modules() if isinstance(mod, GraniteMoeMoE)]

# === 相对 margin PAMS ===
def compute_pams_reg_rel(blocks, eps_rel, k, device):
    regs = []
    for name, blk, _ in blocks:
        if not hasattr(blk, "_last_logits"): continue
        logits = blk._last_logits  # [N, E] fp32, 带梯度
        z_sorted = logits.sort(dim=-1, descending=True).values
        z_k = z_sorted[:, k - 1]
        z_k1 = z_sorted[:, k]
        d_t = z_k - z_k1
        sigma = z_k.abs() + z_k1.abs()  # 尺度归一化分母
        d_t_rel = d_t / sigma.clamp_min(1e-12)
        regs.append(F.relu(eps_rel - d_t_rel).sum())
    if not regs: return torch.zeros((), device=device)
    return sum(regs) / len(regs)

def calibrate_eps_rel(model, ids, k, n_win=50, device="cuda"):
    moe_blocks = discover_moe_blocks(model)
    all_dt_rel = []
    model.eval()
    with torch.no_grad():
        for wi in range(n_win):
            wid = ids[wi * PAMS_STEP:wi * PAMS_STEP + PAMS_WIN].unsqueeze(0).to(device)
            model(**dict(input_ids=wid, attention_mask=torch.ones_like(wid)))
            for _, blk, _ in moe_blocks:
                if hasattr(blk, "_last_captured_logits"):
                    logits = blk._last_captured_logits
                    zs = logits.sort(dim=-1, descending=True).values
                    d_t = zs[:, k-1] - zs[:, k]
                    sigma = zs[:, k-1].abs() + zs[:, k].abs()
                    all_dt_rel.append((d_t / sigma.clamp_min(1e-12)).cpu())
    all_dt_rel = torch.cat(all_dt_rel)
    p1 = float(all_dt_rel.kthvalue(max(1, int(0.01*len(all_dt_rel)))).values)
    p5 = float(all_dt_rel.kthvalue(max(1, int(0.05*len(all_dt_rel)))).values)
    eps = p1 if p1 > 0 else p5
    print(f"[calib] d_t_rel p1={p1:.6f} p5={p5:.6f} eps_rel={eps:.6f}", flush=True)
    return eps, float(all_dt_rel.median())

# === B 口径 (固定 h, gate 精度变) ===
def measure_flip_B(model, ids, moe_blocks, k, n_win=20, device="cuda"):
    """B 口径: 单次 fp32 forward 捕获 h, 离线比 fp32 vs bf16 gate(h)"""
    model.eval()
    n_layers = len(moe_blocks)
    h_all = {i: [] for i in range(n_layers)}
    with torch.no_grad():
        for wi in range(n_win):
            wid = ids[wi * PAMS_STEP:wi * PAMS_STEP + PAMS_WIN].unsqueeze(0).to(device)
            model(**dict(input_ids=wid, attention_mask=torch.ones_like(wid)))
            for i, (_, blk, _) in enumerate(moe_blocks):
                if hasattr(blk, "_last_captured_h"):
                    h_all[i].append(blk._last_captured_h.cpu())
    results = []
    for i, (_, blk, _) in enumerate(moe_blocks):
        hs = torch.cat(h_all[i]).to(device)  # [T, D] fp32 -> GPU
        w = blk.router.layer.weight.detach()  # bf16 [E, D]
        z_fp = F.linear(hs, w.float())  # [T, E] fp32
        z_bf = F.linear(hs.to(torch.bfloat16), w).float()
        fp_topk = z_fp.topk(k, dim=-1).indices
        bf_topk = z_bf.topk(k, dim=-1).indices
        flip_B = float((fp_topk != bf_topk).any(-1).float().mean())
        zs = z_fp.sort(dim=-1, descending=True).values
        d_t = zs[:, k-1] - zs[:, k]
        sigma = zs[:, k-1].abs() + zs[:, k].abs()
        d_t_rel = (d_t / sigma.clamp_min(1e-12)).cpu().numpy()
        results.append({
            "layer": i, "flip_B": flip_B,
            "dt_rel_median": float(np.median(d_t_rel)),
            "dt_rel_p1": float(np.percentile(d_t_rel, 1)),
        })
    model.train()
    return results

def measure_flip_A(model, ids, moe_blocks, k, n_win=20, device="cuda"):
    """A 口径 (全链路 bf16 vs fp32)"""
    model.eval()
    n_layers = len(moe_blocks)
    bf16_logits = {i: [] for i in range(n_layers)}
    h_all = {i: [] for i in range(n_layers)}
    with torch.no_grad():
        for wi in range(n_win):
            wid = ids[wi * PAMS_STEP:wi * PAMS_STEP + PAMS_WIN].unsqueeze(0).to(device)
            model(**dict(input_ids=wid, attention_mask=torch.ones_like(wid)))
            for i, (_, blk, _) in enumerate(moe_blocks):
                if hasattr(blk, "_last_captured_logits"):
                    bf16_logits[i].append(blk._last_captured_logits.cpu())
                    h_all[i].append(blk._last_captured_h.cpu())
    results = []
    fp32_logits = {}
    for i, (_, blk, _) in enumerate(moe_blocks):
        w_fp32 = blk.router.layer.weight.detach().float().cpu()
        hs = torch.cat(h_all[i])
        fp32_logits[i] = F.linear(hs, w_fp32)
    for i in range(n_layers):
        bf = torch.cat(bf16_logits[i])
        fp = fp32_logits[i]
        flip = float((bf.topk(k, dim=-1).indices != fp.topk(k, dim=-1).indices).any(-1).float().mean())
        zs = bf.sort(dim=-1, descending=True).values
        d_t = zs[:, k-1] - zs[:, k]
        sigma = zs[:, k-1].abs() + zs[:, k].abs()
        d_t_rel = (d_t / sigma.clamp_min(1e-12)).numpy()
        dt_pcts = {p: float(np.percentile(d_t.numpy(), p)) for p in [1, 5, 25, 50, 75, 95, 99]}
        results.append({
            "layer": i, "flip_A": flip, "dt_pct": dt_pcts,
            "dt_rel_median": float(np.median(d_t_rel)),
            "dt_rel_p1": float(np.percentile(d_t_rel, 1)),
        })
    model.train()
    return results

def measure_loss(model, ids, n_win=50, device="cuda"):
    model.eval()
    losses = []
    with torch.no_grad():
        for wi in range(0, n_win, PAMS_BS):
            n = min(PAMS_BS, n_win - wi)
            wids = torch.stack([ids[(wi+j)*PAMS_STEP:(wi+j)*PAMS_STEP+PAMS_WIN].to(device) for j in range(n)])
            out = model(**dict(input_ids=wids, attention_mask=torch.ones_like(wids)))
            l = F.cross_entropy(out.logits[:, :-1].float().reshape(-1, out.logits.size(-1)), wids[:, 1:].reshape(-1), reduction="none")
            losses.append(l.mean().item())
    model.train()
    return float(np.mean(losses))

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[granite_pams_rel] λ={PAMS_LAMBDA} steps={PAMS_STEPS} bs={PAMS_BS} lr={PAMS_LR} seed=42", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print(f"[init] creating random Granite-1B-A400M (from scratch)...", flush=True)
    config = AutoConfig.from_pretrained(MODEL_ID)
    from transformers import GraniteMoeForCausalLM
    model = GraniteMoeForCausalLM(config).to(dtype=torch.bfloat16, device=device)
    print(f"[init] params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M", flush=True)

    if PAMS_LAMBDA > 0:
        model.train()
        model.gradient_checkpointing_enable()
        for p in model.parameters(): p.requires_grad = True

    moe_blocks = discover_moe_blocks(model)
    k = config.num_experts_per_tok
    print(f"[init] {len(moe_blocks)} MoE blocks, k={k}", flush=True)

    # Hook router logits
    def make_hook(blk):
        def hook(mod, inp, out):
            h = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            blk._last_captured_h = h.cpu()
            logits = out[-1]
            blk._last_logits = logits
            blk._last_captured_logits = logits.detach().cpu()
        return hook
    handles = [blk.router.register_forward_hook(make_hook(blk)) for _, blk, _ in moe_blocks]

    ids = load_wikitext103(tokenizer)

    global PAMS_EPS_REL
    if PAMS_LAMBDA > 0 and os.environ.get("PAMS_EPS_REL") is None:
        eps_calib, dt_rel_med = calibrate_eps_rel(model, ids, k, device=device)
        PAMS_EPS_REL = eps_calib

    # 训练前 (B + A 双口径measurement)
    print(f"\n=== BEFORE ===", flush=True)
    before_B = measure_flip_B(model, ids[:PAMS_WIN*50], moe_blocks, k, n_win=50, device=device)
    before_A = measure_flip_A(model, ids[:PAMS_WIN*50], moe_blocks, k, n_win=50, device=device)
    before_loss = measure_loss(model, ids, n_win=50, device=device)
    flip_B0 = before_B[0]["flip_B"]
    flip_A0 = before_A[0]["flip_A"]
    print(f"[before] loss={before_loss:.4f} flip_B_L0={flip_B0:.4f} flip_A_L0={flip_A0:.4f}", flush=True)

    if PAMS_LAMBDA == 0 and PAMS_STEPS == 0:
        result = {"model": MODEL_ID, "lambda": 0, "steps": 0, "verdict": "BASELINE_RANDOM_INIT",
            "before": {"flip_B": before_B, "flip_A": before_A, "loss": before_loss},
            "after": {"flip_B": before_B, "flip_A": before_A, "loss": before_loss}}
        with open(OUT_PATH, "w") as f: json.dump(result, f, indent=2)
        print(f"[done] baseline written to {OUT_PATH}", flush=True)
        for h in handles: h.remove(); return

    # 训练
    n_train_ids = min(len(ids), 500000)
    train_starts = list(range(0, (n_train_ids - PAMS_WIN) // PAMS_STEP * PAMS_STEP, PAMS_STEP))
    optimizer = torch.optim.AdamW(model.parameters(), lr=PAMS_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PAMS_STEPS)
    checkpoints, loss_log, pams_log = [], [], []
    t0 = time.time()
    for step in range(PAMS_STEPS):
        start = train_starts[step % len(train_starts)]
        wid = ids[start:start + PAMS_WIN].unsqueeze(0).to(device)
        out = model(**dict(input_ids=wid, attention_mask=torch.ones_like(wid)))
        ce = F.cross_entropy(out.logits[0, :-1].float(), wid[0, 1:])
        pams = compute_pams_reg_rel(moe_blocks, PAMS_EPS_REL, k, device)
        total = ce + PAMS_LAMBDA * pams
        if torch.isnan(total) or torch.isinf(total):
            print(f"[step {step}] NaN/Inf! ce={ce:.4f} pams={pams:.4f}", flush=True); break
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(); scheduler.step()
        loss_log.append(float(ce.detach())); pams_log.append(float(pams.detach()))
        if (step + 1) % 500 == 0 or step == 0:
            print(f"[{step+1}/{PAMS_STEPS}] ce={np.mean(loss_log[-100:]):.4f} pams={np.mean(pams_log[-100:]):.4f} "
                  f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB {time.time()-t0:.0f}s", flush=True)
            flip_B = measure_flip_B(model, ids[:PAMS_WIN*20], moe_blocks, k, n_win=20, device=device)
            checkpoints.append({"step": step+1, "flip_B": flip_B,
                "loss": measure_loss(model, ids, n_win=20, device=device)})

    # 训练后 (B + A 双口径)
    print(f"\n=== AFTER ===", flush=True)
    after_B = measure_flip_B(model, ids[:PAMS_WIN*50], moe_blocks, k, n_win=50, device=device)
    after_A = measure_flip_A(model, ids[:PAMS_WIN*50], moe_blocks, k, n_win=50, device=device)
    after_loss = measure_loss(model, ids, n_win=50, device=device)

    flip_B_avg_after = np.mean([l["flip_B"] for l in after_B])
    flip_A_avg_after = np.mean([l["flip_A"] for l in after_A])
    verdict = "REL_PAMS"
    if flip_B_avg_after <= 0.02:
        verdict += "_B_FLIP_STRONGLY_REDUCED"
    elif flip_B_avg_after <= 0.04:
        verdict += "_B_FLIP_REDUCED"
    else:
        verdict += "_B_FLIP_NOT_REDUCED"

    print(f"[verdict] {verdict}", flush=True)
    print(f"  flip_B_avg={flip_B_avg_after:.4f} flip_A_avg={flip_A_avg_after:.4f} "
          f"dt_rel_med_L0={after_B[0]['dt_rel_median']:.4f} loss={after_loss:.4f}", flush=True)

    result = {"model": MODEL_ID, "lambda": PAMS_LAMBDA, "steps": PAMS_STEPS,
        "eps_rel": PAMS_EPS_REL, "lr": PAMS_LR, "verdict": verdict,
        "before": {"flip_B": before_B, "flip_A": before_A, "loss": before_loss},
        "after": {"flip_B": after_B, "flip_A": after_A, "loss": after_loss},
        "checkpoints": checkpoints}
    with open(OUT_PATH, "w") as f: json.dump(result, f, indent=2)
    print(f"[done] {OUT_PATH}", flush=True)
    for h in handles: h.remove()

if __name__ == "__main__":
    main()
