"""run_olmoe_all.py — OLMoE BEAT 全层三arm (λ=0, 0.1, 1.0, 各 500 steps)
"""
import os, sys, json, time, torch, torch.nn.functional as F
import numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
LOG_DIR = os.path.join(SCRIPT_DIR, "..", "logs")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
MODEL_ID = "allenai/OLMoE-1B-7B-0125"
DEVICE = "cuda"

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "datasets--wikitext", "snapshots", "b08601e04326c79dfdd32d625aee71d232d685c3", "wikitext-103-raw-v1")
LR, STEPS, BS, WIN, STEP, TAU = 5e-5, 500, 1, 128, 64, 0.02
TRAIN_HOLDOUT_OFFSET, VAL_OFFSET = 500000, 600000

def load_data():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    texts = []
    for fn in ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]:
        df = pd.read_parquet(os.path.join(CACHE, fn))
        texts.extend(t for t in df["text"] if isinstance(t, str) and len(t.strip()) > 0)
    all_ids = []
    for i in range(0, len(texts), 1000):
        for ids in tok(texts[i:i+1000], add_special_tokens=False)["input_ids"]:
            all_ids.extend(ids)
    ids = torch.tensor(all_ids, dtype=torch.long)
    return ids, tok

def align_loss(blocks, tau, k, device):
    total, cnt = 0.0, 0
    for _, blk in blocks:
        h = blk._last_captured_h.to(device)
        w = blk.gate.weight.detach().float()
        logits = F.linear(h, w)
        zs = logits.sort(dim=-1, descending=True)
        m = (zs.values[:, k-1] - zs.values[:, k] < tau).cpu()
        if not m.any(): continue
        ek = zs.indices[:, k-1][m]; ek1 = zs.indices[:, k][m]
        hb = h[m]; gu = blk.experts.gate_up_proj; dn = blk.experts.down_proj
        for idx in range(m.sum().item()):
            e_i, e_j = int(ek[idx]), int(ek1[idx])
            hi = hb[idx:idx+1].float()
            yi = F.linear(F.silu(F.linear(hi, gu[e_i].float()).chunk(2,-1)[0]) * F.linear(hi, gu[e_i].float()).chunk(2,-1)[1], dn[e_i].float())
            yj = F.linear(F.silu(F.linear(hi, gu[e_j].float()).chunk(2,-1)[0]) * F.linear(hi, gu[e_j].float()).chunk(2,-1)[1], dn[e_j].float())
            total += (yi-yj).norm()**2 / (yi.norm()**2 + yj.norm()**2 + 1e-8); cnt += 1
    return (total/max(cnt,1)).to(device) if cnt>0 else torch.zeros((), device=device)

def run_arm(lam, tag):
    print(f"\n{'='*60}\n[run] λ={lam} tag={tag}\n{'='*60}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    for p in model.parameters(): p.requires_grad = False
    UNFREEZE_LAYERS = [6, 7, 8, 9]  # 解冻 4 个中间层
    moe_blocks = [(n,m) for n,m in model.named_modules() if isinstance(m, OlmoeSparseMoeBlock)]
    for li, (_, blk) in enumerate(moe_blocks):
        if li in UNFREEZE_LAYERS:
            blk.experts.gate_up_proj.requires_grad = True
            blk.experts.down_proj.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable: {trainable/1e6:.0f}M / {sum(p.numel() for p in model.parameters())/1e6:.0f}M total (layers {UNFREEZE_LAYERS})", flush=True)

    def make_hook(blk):
        def hook(mod, inp, out):
            blk._last_captured_h = inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu()
        return hook
    handles = [b.gate.register_forward_hook(make_hook(b)) for _, b in moe_blocks]

    opt = torch.optim.Adafactor([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    t0 = time.time()
    loss_log = []

    for step in range(STEPS):
        start = (step * STEP) % (len(ids) - WIN)
        wid = ids[start:start+WIN].unsqueeze(0).to(DEVICE)
        out = model(**dict(input_ids=wid, attention_mask=torch.ones_like(wid)))
        ce = F.cross_entropy(out.logits[0, :-1].float(), wid[0, 1:])
        align = align_loss(moe_blocks, TAU, 8, DEVICE)
        total = ce + lam * align
        total.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad(); sch.step()
        loss_log.append(float(ce.detach()))
        if (step+1) % 100 == 0:
            print(f"  [{step+1}/{STEPS}] ce={np.mean(loss_log[-50:]):.4f} align={align:.4f} mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB {time.time()-t0:.0f}s", flush=True)

    # Eval (少窗口)
    model.eval()
    model.config.use_cache = False
    with torch.no_grad():
        ids_val = ids[VAL_OFFSET:VAL_OFFSET+WIN*3]
        vl = []
        for wi in range(3):
            w = ids_val[wi*STEP:wi*STEP+WIN].unsqueeze(0).to(DEVICE)
            o = model(**dict(input_ids=w, attention_mask=torch.ones_like(w)))
            vl.append(F.cross_entropy(o.logits[0,:-1].float(), w[0,1:]).item())
        val_loss = float(np.mean(vl))
        ids_tr = ids[TRAIN_HOLDOUT_OFFSET:TRAIN_HOLDOUT_OFFSET+WIN*3]
        tl = []
        for wi in range(3):
            w = ids_tr[wi*STEP:wi*STEP+WIN].unsqueeze(0).to(DEVICE)
            o = model(**dict(input_ids=w, attention_mask=torch.ones_like(w)))
            tl.append(F.cross_entropy(o.logits[0,:-1].float(), w[0,1:]).item())
        train_loss = float(np.mean(tl))
    print(f"  train_loss={train_loss:.4f} val_loss={val_loss:.4f}", flush=True)

    for h in handles: h.remove()
    result = {"model": MODEL_ID, "lambda": lam, "steps": STEPS, "trainable_M": trainable/1e6,
        "train_loss": train_loss, "val_loss": val_loss, "ce_trace": loss_log}
    with open(os.path.join(OUT_DIR, f"olmoe_beat_all_{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [done] olmoe_beat_all_{tag}.json", flush=True)

# Load data once
ids, _ = load_data()
print(f"[data] {len(ids)} tokens", flush=True)
for lam, tag in [(1.0, "l10")]:
    run_arm(lam, tag)
print("\n[run] All done", flush=True)
