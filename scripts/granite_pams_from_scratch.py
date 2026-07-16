"""granite_pams_from_scratch.py — 格A: Granite-1B-A400M from-scratch PAMS 训练

在 Granite-3.1-1B-A400M 架构上从随机初始化训练，verification PAMS 能否降翻转推 margin。
架构: 1.3B total / 400M active / 32 experts / k=8 / 24层全MoE

design:
  - loading Granite config → 创建随机初始化模型 (不loading预训练权重)
  - Wikitext-103 128-token 窗训练
  - PAMS regularization: λ·Σ relu(ε - d_t), d_t = z[k-1] - z[k]
  - 每 N steps checkpoint measurement flip_A + dt_med + loss
  - λ sweep: 0 (baseline), 0.01, 0.1, 1.0, 3.0

Granite MoE 特殊处理:
  - moe_layer = model.model.layers[i].block_sparse_moe
  - moe_layer.router = GraniteMoeTopKGating, 含 .layer.weight (gate)
  - moe_layer.router.forward(h) 返回 (..., logits), logits=[N, E] 原始softmax前
  - 无 .gate/.experts 属性 → 不走 mtp adapter, 直接操作 router

环境: PAMS_LAMBDA (默认 0, baseline), PAMS_STEPS (默认 5000),
  PAMS_EPS (默认 ε 标定值), PAMS_LR (默认 3e-4), PAMS_BS (默认 4)
output: results/granite_pams_l{LAMBDA}.json
"""
import os, sys, json, gc, time, math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeMoE, GraniteMoeTopKGating

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-base"

PAMS_LAMBDA = float(os.environ.get("PAMS_LAMBDA", "0"))
PAMS_STEPS = int(os.environ.get("PAMS_STEPS", "5000"))
PAMS_EPS = float(os.environ.get("PAMS_EPS", "0.02"))
PAMS_LR = float(os.environ.get("PAMS_LR", "3e-4"))
PAMS_BS = int(os.environ.get("PAMS_BS", "4"))
PAMS_WIN = 128
PAMS_STEP = 64
PAMS_WARMUP = 500

OUT_TAG = os.environ.get("PAMS_TAG", f"l{str(PAMS_LAMBDA).replace('.','_')}")
OUT_PATH = os.path.join(OUT_DIR, f"granite_pams_{OUT_TAG}.json")

# === Data: Wikitext-103 parquet ===
DATA_CACHE = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "datasets--wikitext", "snapshots", "b08601e04326c79dfdd32d625aee71d232d685c3",
    "wikitext-103-raw-v1",
)

def load_wikitext103(tokenizer, max_examples=None):
    import pandas as pd
    texts = []
    for fn in ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]:
        fp = os.path.join(DATA_CACHE, fn)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"missing {fp}")
        df = pd.read_parquet(fp)
        for text in df["text"]:
            if isinstance(text, str) and len(text.strip()) > 0:
                texts.append(text)
                if max_examples and len(texts) >= max_examples:
                    break
        if max_examples and len(texts) >= max_examples:
            break
    # 分批 tokenize 避免 OOM
    all_ids = []
    for i in range(0, len(texts), 1000):
        batch = texts[i:i+1000]
        enc = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for ids in enc:
            all_ids.extend(ids)
    ids = torch.tensor(all_ids, dtype=torch.long)
    print(f"[data] {len(ids)} tokens ({len(texts)} lines)", flush=True)
    return ids


# === MoE block discovery for Granite ===
def discover_granite_moe_blocks(model):
    blocks = []
    for name, mod in model.named_modules():
        if isinstance(mod, GraniteMoeMoE):
            blocks.append((name, mod, mod.router.num_experts))
    return blocks


def compute_pams_reg(blocks, eps, k, device):
    # block forward 后从 router.forward 抓 logits
    regs = []
    for name, blk, _ in blocks:
        if not hasattr(blk, "_last_logits"):
            continue
        logits = blk._last_logits  # [N, E] fp32, 带梯度
        z_sorted = logits.sort(dim=-1, descending=True).values
        d_t = z_sorted[:, k - 1] - z_sorted[:, k]
        regs.append(F.relu(eps - d_t).sum())
    if not regs:
        return torch.zeros((), device=device)
    return sum(regs) / len(regs)


def calibrate_eps(model, ids, tokenizer, k, n_win=50, device="cuda"):
    """Step0: measurement d_t 1% 分位定 ε"""
    print("[calib] measuring d_t distribution for ε calibration...", flush=True)
    moe_blocks = discover_granite_moe_blocks(model)
    all_dt = []
    model.eval()
    with torch.no_grad():
        for wi in range(n_win):
            wid = ids[wi * PAMS_STEP:wi * PAMS_STEP + PAMS_WIN].unsqueeze(0).to(device)
            enc = {"input_ids": wid, "attention_mask": torch.ones_like(wid)}
            model(**enc)
            for _, blk, _ in moe_blocks:
                if hasattr(blk, "_last_captured_logits"):
                    logits = blk._last_captured_logits
                    z_sorted = logits.sort(dim=-1, descending=True).values
                    d_t = z_sorted[:, k - 1] - z_sorted[:, k]
                    all_dt.append(d_t.cpu())
    all_dt = torch.cat(all_dt)
    dt_p1 = float(all_dt.kthvalue(max(1, int(0.01 * len(all_dt)))).values)
    dt_p5 = float(all_dt.kthvalue(max(1, int(0.05 * len(all_dt)))).values)
    eps = dt_p1 if dt_p1 > 0 else dt_p5
    print(f"[calib] d_t p1={dt_p1:.6f} p5={dt_p5:.6f} eps={eps:.6f}", flush=True)
    return eps, float(all_dt.median())


# === A 口径measurement (flip_A = bf16 vs fp32 gate top-k diff) ===
def measure_flip_A(model, ids, moe_blocks, k, n_win=20, device="cuda"):
    """measurement每层 flip_A + d_t 分布. 一次 forward 收集 bf16 logits + h, 再离线算 fp32 logits."""
    model.eval()
    n_layers = len(moe_blocks)
    # 收集所有 windows 的 bf16 logits 和 h
    bf16_logits = {i: [] for i in range(n_layers)}
    h_all = {i: [] for i in range(n_layers)}
    with torch.no_grad():
        for wi in range(n_win):
            wid = ids[wi * PAMS_STEP:wi * PAMS_STEP + PAMS_WIN].unsqueeze(0).to(device)
            enc = {"input_ids": wid, "attention_mask": torch.ones_like(wid)}
            model(**enc)
            for i, (_, blk, _) in enumerate(moe_blocks):
                if hasattr(blk, "_last_captured_logits"):
                    bf16_logits[i].append(blk._last_captured_logits.cpu())
                    h_all[i].append(blk._last_captured_h.cpu())
    results = []
    # 离线算 fp32 logits (gate.weight 转 fp32 后 linear)
    fp32_logits = {}
    for i, (_, blk, _) in enumerate(moe_blocks):
        w_fp32 = blk.router.layer.weight.detach().float().cpu()  # [E, D]
        hs = torch.cat(h_all[i])  # [T, D] fp32 cpu
        fp32_logits[i] = F.linear(hs, w_fp32)
    for i in range(len(moe_blocks)):
        bf = torch.cat(bf16_logits[i])  # [T, E]
        fp = fp32_logits[i]
        bf_topk = bf.topk(k, dim=-1).indices
        fp_topk = fp.topk(k, dim=-1).indices
        flip_rate = float((bf_topk != fp_topk).any(-1).float().mean())
        # d_t distribution from bf16
        z_sorted = bf.sort(dim=-1, descending=True).values
        d_t = z_sorted[:, k - 1] - z_sorted[:, k]
        dt_pcts = {p: float(np.percentile(d_t.numpy(), p)) for p in [1, 5, 25, 50, 75, 95, 99]}
        frac_below = float((d_t < PAMS_EPS).float().mean())
        results.append({
            "layer": i, "flip_A": flip_rate, "dt_pct": dt_pcts,
            "frac_dt_below_eps": frac_below,
            "n_tokens": int(bf.shape[0]),
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
            enc = {"input_ids": wids, "attention_mask": torch.ones_like(wids)}
            out = model(**enc)
            logits = out.logits[:, :-1].float()
            targets = wids[:, 1:]
            l = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
            losses.append(l.mean().item())
    model.train()
    return float(np.mean(losses))


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[granite_pams] λ={PAMS_LAMBDA} steps={PAMS_STEPS} bs={PAMS_BS} lr={PAMS_LR} seed=42", flush=True)
    
    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 随机初始化模型 (不loading预训练权重!)
    print(f"[init] creating random Granite-1B-A400M (from scratch)...", flush=True)
    config = AutoConfig.from_pretrained(MODEL_ID)
    from transformers import GraniteMoeForCausalLM
    model = GraniteMoeForCausalLM(config)
    model = model.to(dtype=torch.bfloat16, device=device)
    print(f"[init] params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M", flush=True)
    
    if PAMS_LAMBDA > 0:
        model.train()
        model.gradient_checkpointing_enable()
        for p in model.parameters():
            p.requires_grad = True
        print(f"[init] trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.0f}M params", flush=True)
    
    # 发现 MoE blocks
    moe_blocks = discover_granite_moe_blocks(model)
    k = config.num_experts_per_tok
    print(f"[init] {len(moe_blocks)} MoE blocks, k={k}", flush=True)
    
    # 安装 hook — 抓 router.logits (带梯度用于 PAMS) + h (detach fp32, 用于measurement)
    _CAPTURED_LOGITS = []
    _CAPTURED_H = []
    def make_logits_hook(blk):
        def hook(mod, inp, out):
            h = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            blk._last_captured_h = h.cpu()
            logits = out[-1]  # router 返回 (index_sorted, batch_index, batch_gates, expert_size, logits)
            blk._last_logits = logits  # 带梯度
            blk._last_captured_logits = logits.detach().cpu()
        return hook
    handles = []
    for _, blk, _ in moe_blocks:
        h = blk.router.register_forward_hook(make_logits_hook(blk))
        handles.append(h)
    
    # loading数据
    print(f"[data] loading Wikitext-103...", flush=True)
    ids = load_wikitext103(tokenizer)
    
    # ε 标定
    global PAMS_EPS
    if PAMS_LAMBDA > 0 and os.environ.get("PAMS_EPS") is None:
        eps_calib, dt_med = calibrate_eps(model, ids, tokenizer, k, device=device)
        PAMS_EPS = eps_calib
        print(f"[ε] calibrated ε={PAMS_EPS:.6f} (d_t median={dt_med:.6f})", flush=True)
    
    # 训练循环
    n_train_ids = min(len(ids), 500000)  # 用 500K tokens 子集
    train_ids = ids[:n_train_ids]
    n_windows = (n_train_ids - PAMS_WIN) // PAMS_STEP
    train_starts = list(range(0, n_windows * PAMS_STEP, PAMS_STEP))
    print(f"[train] {n_windows} windows from {n_train_ids} tokens", flush=True)
    
    # before measurement
    print(f"\n=== BEFORE ===", flush=True)
    before_flip = measure_flip_A(model, ids[:PAMS_WIN*50], moe_blocks, k, n_win=50, device=device)
    before_loss = measure_loss(model, ids, n_win=50, device=device)
    print(f"[before] loss={before_loss:.4f}", flush=True)
    for r in before_flip[:3]:
        print(f"  L{r['layer']}: flip_A={r['flip_A']:.4f} dt_med={r['dt_pct'][50]:.4f}", flush=True)
    
    if PAMS_LAMBDA == 0 and PAMS_STEPS == 0:
        # baseline: 不训练, 直接output随机初始化measurement
        model.eval()
        after_flip = measure_flip_A(model, ids[:PAMS_WIN*50], moe_blocks, k, n_win=50, device=device)
        after_loss = measure_loss(model, ids, n_win=50, device=device)
        result = {
            "model": MODEL_ID, "lambda": 0, "steps": 0,
            "verdict": "BASELINE_RANDOM_INIT",
            "before": {"flip": before_flip, "loss": before_loss},
            "after": {"flip": after_flip, "loss": after_loss},
            "checkpoints": [],
        }
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[done] baseline written to {OUT_PATH}", flush=True)
        for h in handles: h.remove()
        return
    
    # λ=0 但 steps>0: 训练baseline (无 PAMS, hard routing)
    if PAMS_LAMBDA == 0:
        print(f"\n=== HARD TRAINING (λ=0, {PAMS_STEPS} steps) ===", flush=True)
    
    # PAMS 训练
    optimizer = torch.optim.AdamW(model.parameters(), lr=PAMS_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PAMS_STEPS)
    checkpoints = []
    loss_log = []
    pams_log = []
    print(f"\n=== PAMS TRAINING (λ={PAMS_LAMBDA}) ===", flush=True)
    t0 = time.time()
    for step in range(PAMS_STEPS):
        start = train_starts[step % len(train_starts)]
        wid = ids[start:start + PAMS_WIN].unsqueeze(0).to(device)
        enc = {"input_ids": wid, "attention_mask": torch.ones_like(wid)}
        out = model(**enc)
        ce = F.cross_entropy(out.logits[0, :-1].float(), wid[0, 1:])
        pams = compute_pams_reg(moe_blocks, PAMS_EPS, k, device)
        total = ce + PAMS_LAMBDA * pams
        if torch.isnan(total) or torch.isinf(total):
            print(f"[step {step}] NaN/Inf! ce={ce:.4f} pams={pams:.4f}", flush=True)
            break
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        loss_log.append(float(ce.detach()))
        pams_log.append(float(pams.detach()))
        if (step + 1) % 500 == 0 or step == 0:
            loss_last = float(np.mean(loss_log[-100:]))
            pams_last = float(np.mean(pams_log[-100:]))
            mem = torch.cuda.max_memory_allocated() / 1e9
            now = time.time()
            print(f"[{step+1}/{PAMS_STEPS}] ce={loss_last:.4f} pams={pams_last:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                  f"mem={mem:.1f}GB {now-t0:.0f}s", flush=True)
            # checkpoint measurement
            flip = measure_flip_A(model, ids[:PAMS_WIN*20], moe_blocks, k, n_win=20, device=device)
            closs = measure_loss(model, ids, n_win=20, device=device)
            checkpoints.append({
                "step": step + 1, "loss": closs, "flip": flip,
            })
    
    # after measurement
    print(f"\n=== AFTER ===", flush=True)
    after_flip = measure_flip_A(model, ids[:PAMS_WIN*50], moe_blocks, k, n_win=50, device=device)
    after_loss = measure_loss(model, ids, n_win=50, device=device)
    
    verdict = "PAMS_ON_GRANITE"
    if after_flip[0]["flip_A"] < before_flip[0]["flip_A"] * 0.8:
        verdict += "_FLIP_REDUCED"
    else:
        verdict += "_FLIP_NOT_REDUCED"
    print(f"[verdict] {verdict}", flush=True)
    for r in after_flip[:3]:
        print(f"  L{r['layer']}: flip_A={r['flip_A']:.4f} dt_med={r['dt_pct'][50]:.4f}", flush=True)
    
    result = {
        "model": MODEL_ID, "lambda": PAMS_LAMBDA, "steps": PAMS_STEPS,
        "eps": PAMS_EPS, "lr": PAMS_LR, "bs": PAMS_BS,
        "verdict": verdict,
        "before": {"flip": before_flip, "loss": before_loss},
        "after": {"flip": after_flip, "loss": after_loss},
        "checkpoints": checkpoints,
        "loss_log": loss_log[-500:],
        "pams_log": pams_log[-500:],
    }
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] {OUT_PATH}", flush=True)
    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
