"""granite_pams_align.py — boundaryalignmentregularization (定律训练侧第二发)

regularization: L_align = λ · Σ ‖e_i - e_j‖²/(‖e_i‖²+‖e_j‖²)
  i=z[k], j=z[k+1] (竞争expert对), 仅boundary token (d_t < τ)
  router logits detach (gate 不参与梯度)

主终点: M2 (tear_median) 逐层decrease
副终点: CE 非劣, flip_B/A 预期unchanged

环境: PAMS_LAMBDA (alignment权重, 默认 0.1), PAMS_TAU (boundary阈值, 默认 0.02),
  PAMS_STEPS, PAMS_LR
output: results/granite_pams_align_l{LAMBDA}.json
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeMoE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-base"

PAMS_LAMBDA = float(os.environ.get("PAMS_LAMBDA", "0.1"))
PAMS_TAU = float(os.environ.get("PAMS_TAU", "0.02"))
PAMS_STEPS = int(os.environ.get("PAMS_STEPS", "5000"))
PAMS_LR = float(os.environ.get("PAMS_LR", "3e-4"))
PAMS_BS = int(os.environ.get("PAMS_BS", "4"))
PAMS_SEED = int(os.environ.get("PAMS_SEED", "42"))
PAMS_WIN = 128
PAMS_STEP = 64
PAMS_CKPT = int(os.environ.get("PAMS_CKPT", "500"))  # checkpoint interval
VAL_OFFSET = 600000  # validation slice offset (beyond training's first 500k tokens)
TRAIN_HOLDOUT_OFFSET = 500000  # training eval offset (right after training's 500k, non-overlapping)

OUT_TAG = os.environ.get("PAMS_TAG", f"l{str(PAMS_LAMBDA).replace('.','_')}_s{PAMS_SEED}")
OUT_PATH = os.path.join(OUT_DIR, f"granite_pams_align_{OUT_TAG}.json")
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "..", "weights")
BEST_WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "..", "weights_best")

# === Data ===
DATA_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "datasets--wikitext", "snapshots", "b08601e04326c79dfdd32d625aee71d232d685c3",
    "wikitext-103-raw-v1")

def load_wikitext103(tokenizer, max_examples=None, offset=0, max_tokens=None):
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
    if offset > 0:
        ids = ids[offset:]
    if max_tokens is not None:
        ids = ids[:max_tokens]
    print(f"[data] {len(ids)} tokens ({len(texts)} lines, offset={offset})", flush=True)
    return ids

def discover_moe_blocks(model):
    return [(name, mod, mod.router.num_experts) for name, mod in model.named_modules() if isinstance(mod, GraniteMoeMoE)]

# === 单expert forward (Granite fused format) ===
def expert_forward_granite(blk, e, h):
    w_gu = blk.input_linear.weight[e].float()   # [D, 2*H]
    w_dn = blk.output_linear.weight[e].float()  # [H, D]
    hidden = F.linear(h, w_gu)
    gate, up = hidden.chunk(2, dim=-1)
    act = blk.activation(gate) * up
    return F.linear(act, w_dn)

# === boundaryalignmentregularization ===
def compute_align_loss(blocks, tau, k, device):
    total, count = 0.0, 0
    for name, blk, _ in blocks:
        if not hasattr(blk, "_last_logits") or not hasattr(blk, "_last_captured_h"): continue
        logits = blk._last_logits.detach()  # 停梯度: router 不参与alignmentregularization
        zs = logits.sort(dim=-1, descending=True)
        d_t = zs.values[:, k-1] - zs.values[:, k]
        mask = (d_t < tau).cpu()  # boundary token
        if not mask.any(): continue
        ek = zs.indices[:, k-1][mask]   # 第 k 名 (cpu)
        ek1 = zs.indices[:, k][mask]    # 第 k+1 名 (cpu)
        h_b = blk._last_captured_h[mask].to(device)  # [B, D] fp32

        # batch 版本: 逐 token 计算 (boundary token 少, 循环可接受)
        for idx in range(mask.sum().item()):
            e_i, e_j = int(ek[idx]), int(ek1[idx])
            yi = expert_forward_granite(blk, e_i, h_b[idx:idx+1])
            yj = expert_forward_granite(blk, e_j, h_b[idx:idx+1])
            yi = yi.reshape(-1); yj = yj.reshape(-1)
            num = (yi - yj).norm()**2
            den = yi.norm()**2 + yj.norm()**2 + 1e-8
            total += num / den
            count += 1
    return (total / max(count, 1)).to(device) if count > 0 else torch.zeros((), device=device)

# === 增强 M2 measurement: cos per margin 桶 + M1 分布 + hardG ===
def measure_all(model, ids, moe_blocks, k, n_win=50, device="cuda"):
    """返回 {layer_i: {tear_median, cos_median, cos_buckets, m1_pcts, hardG}}"""
    model.eval()
    n_layers = len(moe_blocks)
    all_h = {i: [] for i in range(n_layers)}
    all_ek = {i: [] for i in range(n_layers)}
    all_ek1 = {i: [] for i in range(n_layers)}
    all_topk = {i: [] for i in range(n_layers)}
    all_dt = {i: [] for i in range(n_layers)}
    with torch.no_grad():
        for wi in range(n_win):
            wid = ids[wi*PAMS_STEP:wi*PAMS_STEP+PAMS_WIN].unsqueeze(0).to(device)
            model(**dict(input_ids=wid, attention_mask=torch.ones_like(wid)))
            for i, (_, blk, _) in enumerate(moe_blocks):
                if hasattr(blk,"_last_captured_h") and hasattr(blk,"_last_captured_logits"):
                    all_h[i].append(blk._last_captured_h)
                    logits = blk._last_captured_logits  # [T, E] cpu
                    zs = logits.sort(dim=-1, descending=True)
                    all_ek[i].append(zs.indices[:, k-1])
                    all_ek1[i].append(zs.indices[:, k])
                    all_topk[i].append(zs.indices[:, :k])
                    d_t = zs.values[:, k-1] - zs.values[:, k]
                    all_dt[i].append(d_t)
    results = []
    for i, (_, blk, _) in enumerate(moe_blocks):
        h = torch.cat(all_h[i]).to(device)
        ek = torch.cat(all_ek[i])
        ek1 = torch.cat(all_ek1[i])
        topk_all = torch.cat(all_topk[i])  # [T, k]
        d_t_all = torch.cat(all_dt[i]).numpy()  # [T]
        # 算竞争expertoutput
        yk, yk1 = torch.zeros_like(h), torch.zeros_like(h)
        for e_idx in range(blk.router.num_experts):
            m_k = (ek == e_idx); m_k1 = (ek1 == e_idx)
            if m_k.any(): yk[m_k] = expert_forward_granite(blk, e_idx, h[m_k])
            if m_k1.any(): yk1[m_k1] = expert_forward_granite(blk, e_idx, h[m_k1])
        num = (yk - yk1).norm(dim=-1); den = yk.norm(dim=-1) + yk1.norm(dim=-1) + 1e-6
        tear = (num / den).detach().cpu().numpy()
        cos_raw = ((yk * yk1).sum(dim=-1) / (yk.norm(dim=-1) * yk1.norm(dim=-1) + 1e-6)).detach().cpu().numpy()
        # ① Cos per margin 桶
        edges = [0, 0.01, 0.05, 0.2, float('inf')]
        labels = ["lt0.01", "0.01-0.05", "0.05-0.2", "gt0.2"]
        cos_buckets = {}
        for lo, hi, lb in zip(edges[:-1], edges[1:], labels):
            if hi == float('inf'):
                m = d_t_all >= lo
            else:
                m = (d_t_all >= lo) & (d_t_all < hi)
            vals = cos_raw[m]
            cos_buckets[lb] = {
                "n": int(m.sum()),
                "cos_median": float(np.nanmedian(vals)) if len(vals) > 0 else None,
            }
        # ② M1 分布
        n_boundary = int((d_t_all < 0.05).sum())
        m1_pcts = {str(p): float(np.percentile(d_t_all, p)) for p in [1,5,25,50,75,90,95,99]}
        T = len(h)
        # ③ 真拓扑: expert利用率 (unique experts in top-k / total experts) + 负载熵
        flat = topk_all.numpy().flatten()  # [T*k]
        unique_experts = len(set(int(x) for x in flat))
        expert_util = unique_experts / blk.router.num_experts
        # 负载熵: -Σ(p_e × ln p_e), p_e = expert e 在 top-k 中出现的频率
        expert_counts = np.bincount([int(x) for x in flat], minlength=blk.router.num_experts)
        p_e = expert_counts / max(expert_counts.sum(), 1)
        load_entropy = float(-(p_e[p_e > 0] * np.log(p_e[p_e > 0])).sum())
        # ④ 全局 pairwise cos (采样 32 个锚点 token 算 32 expert全对)
        n_anchors = min(32, len(h))
        anchor_idx = torch.randperm(len(h))[:n_anchors].to(device)
        all_e_out = []
        for e_idx in range(blk.router.num_experts):
            out = expert_forward_granite(blk, e_idx, h[anchor_idx])
            all_e_out.append(out.detach().cpu())  # [n_anchors, D]
        all_e_out = torch.stack(all_e_out)  # [E, n_anchors, D]
        all_e_out = all_e_out.reshape(all_e_out.shape[0], -1)  # [E, n_anchors*D]
        cos_matrix = torch.mm(F.normalize(all_e_out, dim=-1), F.normalize(all_e_out, dim=-1).T)
        tri = torch.triu(cos_matrix, diagonal=1)
        nonzero = tri[tri > -2]
        avg_pairwise_cos = float(nonzero.mean()) if nonzero.numel() > 0 else 0.0
        results.append({
            "layer": i, "n_tokens": T,
            "tear_median": float(np.nanmedian(tear)),
            "tear_p90": float(np.nanpercentile(tear, 90)),
            "cos_median": float(np.nanmedian(cos_raw)),
            "cos_buckets": cos_buckets,
            "m1_pcts": m1_pcts,
            "boundary_frac_lt0.05": float(n_boundary / T) if T > 0 else 0.0,
            "expert_util": float(expert_util),
            "load_entropy": load_entropy,
            "pairwise_cos_avg": avg_pairwise_cos,
        })
    model.train()
    return results

# === B 口径 (固定 h, gate 精度变) ===
def measure_flip_B(model, ids, moe_blocks, k, n_win=20, device="cuda"):
    model.eval()
    h_all = {i: [] for i in range(len(moe_blocks))}
    with torch.no_grad():
        for wi in range(n_win):
            wid = ids[wi*PAMS_STEP:wi*PAMS_STEP+PAMS_WIN].unsqueeze(0).to(device)
            model(**dict(input_ids=wid, attention_mask=torch.ones_like(wid)))
            for i, (_, blk, _) in enumerate(moe_blocks):
                if hasattr(blk, "_last_captured_h"):
                    h_all[i].append(blk._last_captured_h.cpu())
    results = []
    for i, (_, blk, _) in enumerate(moe_blocks):
        hs = torch.cat(h_all[i]).to(device)
        w = blk.router.layer.weight.detach()
        z_fp = F.linear(hs, w.float())
        z_bf = F.linear(hs.to(torch.bfloat16), w).float()
        flip = float((z_fp.topk(k,dim=-1).indices != z_bf.topk(k,dim=-1).indices).any(-1).float().mean())
        results.append({"layer": i, "flip_B": flip})
    model.train()
    return results

def measure_flip_A(model, ids, moe_blocks, k, n_win=20, device="cuda"):
    model.eval()
    n = len(moe_blocks)
    bf = {i:[] for i in range(n)}; ha = {i:[] for i in range(n)}
    with torch.no_grad():
        for wi in range(n_win):
            wid = ids[wi*PAMS_STEP:wi*PAMS_STEP+PAMS_WIN].unsqueeze(0).to(device)
            model(**dict(input_ids=wid, attention_mask=torch.ones_like(wid)))
            for i, (_, blk, _) in enumerate(moe_blocks):
                if hasattr(blk,"_last_captured_logits"):
                    bf[i].append(blk._last_captured_logits.cpu())
                    ha[i].append(blk._last_captured_h.cpu())
    res = []
    for i, (_, blk, _) in enumerate(moe_blocks):
        w_fp = blk.router.layer.weight.detach().float().cpu()
        hs = torch.cat(ha[i])
        z_fp = F.linear(hs, w_fp)
        z_bf = torch.cat(bf[i])
        flip = float((z_bf.topk(k,dim=-1).indices != z_fp.topk(k,dim=-1).indices).any(-1).float().mean())
        res.append({"layer": i, "flip_A": flip})
    model.train()
    return res

def measure_loss(model, ids, n_win=50, device="cuda"):
    model.eval(); losses = []
    with torch.no_grad():
        for wi in range(0, n_win, PAMS_BS):
            n = min(PAMS_BS, n_win-wi)
            wids = torch.stack([ids[(wi+j)*PAMS_STEP:(wi+j)*PAMS_STEP+PAMS_WIN].to(device) for j in range(n)])
            out = model(**dict(input_ids=wids, attention_mask=torch.ones_like(wids)))
            l = F.cross_entropy(out.logits[:,:-1].float().reshape(-1,out.logits.size(-1)), wids[:,1:].reshape(-1), reduction="none")
            losses.append(l.mean().item())
    model.train()
    return float(np.mean(losses))

def main():
    torch.manual_seed(PAMS_SEED); np.random.seed(PAMS_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tau = PAMS_TAU
    print(f"[granite_pams_align] λ={PAMS_LAMBDA} τ={tau} steps={PAMS_STEPS} bs={PAMS_BS} lr={PAMS_LR} seed={PAMS_SEED}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print(f"[init] creating random Granite-1B-A400M (from scratch)...", flush=True)
    config = AutoConfig.from_pretrained(MODEL_ID)
    from transformers import GraniteMoeForCausalLM
    model = GraniteMoeForCausalLM(config).to(dtype=torch.bfloat16, device=device)
    print(f"[init] params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M", flush=True)

    model.train()
    model.gradient_checkpointing_enable()
    for p in model.parameters(): p.requires_grad = True

    moe_blocks = discover_moe_blocks(model)
    k = config.num_experts_per_tok
    print(f"[init] {len(moe_blocks)} MoE blocks, k={k}", flush=True)

    # Hook: 抓 h (detach fp32) + logits (detach for align reg, 无梯度)
    def make_hook(blk):
        def hook(mod, inp, out):
            h = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            blk._last_captured_h = h.cpu()
            blk._last_captured_logits = out[-1].detach().cpu()
            blk._last_logits = out[-1]  # 带梯度 (但 compute_align_loss 会 detach)
        return hook
    handles = [blk.router.register_forward_hook(make_hook(blk)) for _, blk, _ in moe_blocks]

    ids = load_wikitext103(tokenizer)
    ids_val = ids[VAL_OFFSET:VAL_OFFSET+PAMS_WIN*50]
    ids_train_eval = ids[TRAIN_HOLDOUT_OFFSET:TRAIN_HOLDOUT_OFFSET+PAMS_WIN*50]
    best_val_loss = float('inf')

    # measurement前 (训练 loss = 训练集 holdout, val_loss = verification集)
    print(f"\n=== BEFORE ===", flush=True)
    before_all = measure_all(model, ids_val, moe_blocks, k, n_win=50, device=device)
    before_B = measure_flip_B(model, ids_val, moe_blocks, k, n_win=50, device=device)
    before_train = measure_loss(model, ids_train_eval, n_win=50, device=device)
    before_val = measure_loss(model, ids_val, n_win=50, device=device)
    M2_before = np.mean([l["tear_median"] for l in before_all if not np.isnan(l["tear_median"])])
    print(f"[before] train_loss={before_train:.4f} val_loss={before_val:.4f} M2_avg={M2_before:.4f} "
          f"flip_B_avg={np.mean([l['flip_B'] for l in before_B]):.4f}", flush=True)
    for r in before_all[:5]:
        print(f"  L{r['layer']}: tear={r['tear_median']:.4f} cos={r['cos_median']:.4f} "
              f"bnd={r['boundary_frac_lt0.05']:.3f} util={r['expert_util']:.3f}", flush=True)

    # 训练
    n_train_ids = min(len(ids), 500000)
    train_starts = list(range(0, (n_train_ids-PAMS_WIN)//PAMS_STEP*PAMS_STEP, PAMS_STEP))
    optimizer = torch.optim.AdamW(model.parameters(), lr=PAMS_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PAMS_STEPS)
    checkpoints, loss_log = [], []
    t0 = time.time()
    for step in range(PAMS_STEPS):
        start = train_starts[step % len(train_starts)]
        wid = ids[start:start+PAMS_WIN].unsqueeze(0).to(device)
        out = model(**dict(input_ids=wid, attention_mask=torch.ones_like(wid)))
        ce = F.cross_entropy(out.logits[0, :-1].float(), wid[0, 1:])
        align = compute_align_loss(moe_blocks, tau, k, device)
        total = ce + PAMS_LAMBDA * align
        if torch.isnan(total) or torch.isinf(total):
            print(f"[step {step}] NaN/Inf! ce={ce:.4f} align={align:.4f}", flush=True); break
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(); scheduler.step()
        loss_log.append(float(ce.detach()))
        if (step+1) % 500 == 0 or step == 0:
            ck_val_loss = measure_loss(model, ids_val, n_win=50, device=device)
            if ck_val_loss < best_val_loss:
                best_val_loss = ck_val_loss
                os.makedirs(BEST_WEIGHTS_DIR, exist_ok=True)
                best_path = os.path.join(BEST_WEIGHTS_DIR, f"granite_pams_align_{OUT_TAG}")
                model.save_pretrained(best_path, safe_serialization=True)
                print(f"  [best val_loss] {ck_val_loss:.4f} saved", flush=True)
            print(f"[{step+1}/{PAMS_STEPS}] ce={np.mean(loss_log[-100:]):.4f} "
                  f"align={float(align.detach()):.4f} "
                  f"val_loss={ck_val_loss:.4f} "
                  f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB {time.time()-t0:.0f}s", flush=True)
            ck_all = measure_all(model, ids_val[:PAMS_WIN*20], moe_blocks, k, n_win=20, device=device)
            M2_ck_avg = np.mean([l["tear_median"] for l in ck_all if not np.isnan(l["tear_median"])])
            ut_avg = np.mean([l["expert_util"] for l in ck_all])
            print(f"  M2={M2_ck_avg:.4f} util={ut_avg:.3f} ent={np.mean([l['load_entropy'] for l in ck_all]):.3f}", flush=True)
            checkpoints.append({"step": step+1, "ce": float(ce.detach()), "val_loss": ck_val_loss, "measures": ck_all})

    # measurement后 (训练 loss = 近期训练均值, val_loss = verification集)
    print(f"\n=== AFTER ===", flush=True)
    after_all = measure_all(model, ids_val, moe_blocks, k, n_win=50, device=device)
    after_B = measure_flip_B(model, ids_val, moe_blocks, k, n_win=50, device=device)
    after_A = measure_flip_A(model, ids_val, moe_blocks, k, n_win=50, device=device)
    after_train = float(np.mean(loss_log[-100:])) if len(loss_log) >= 100 else float(np.mean(loss_log))
    after_val = measure_loss(model, ids_val, n_win=50, device=device)

    M2_after = np.mean([l["tear_median"] for l in after_all if not np.isnan(l["tear_median"])])
    M2_reduction = (M2_before - M2_after) / M2_before if M2_before > 0 else 0
    util_after = np.mean([l["expert_util"] for l in after_all])
    util_before = np.mean([l["expert_util"] for l in before_all])
    ent_after = np.mean([l["load_entropy"] for l in after_all])
    flip_B_after = np.mean([l["flip_B"] for l in after_B])
    flip_A_after = np.mean([l["flip_A"] for l in after_A])

    # 四检判定
    avg_pairwise_cos_val = np.mean([l["pairwise_cos_avg"] for l in after_all])
    expert_collapse_flag = avg_pairwise_cos_val > 0.8
    util_drift = abs(util_after - util_before) > 0.15
    if M2_reduction > 0.2 and after_val <= before_val * 1.05 and not expert_collapse_flag:
        verdict = "ALIGN_REDUCES_TEAR_CE_HELD"
    elif M2_reduction > 0.2 and not expert_collapse_flag:
        verdict = "ALIGN_REDUCES_TEAR_CE_INCREASED"
    elif M2_reduction > 0.2 and expert_collapse_flag:
        verdict = "ALIGN_REDUCES_TEAR_BUT_EXPERT_COLLAPSE"
    else:
        verdict = "ALIGN_NOT_REDUCED"

    # saving最终权重
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    wt_path = os.path.join(WEIGHTS_DIR, f"granite_pams_align_{OUT_TAG}")
    model.save_pretrained(wt_path, safe_serialization=True)
    wt_size = sum(os.path.getsize(os.path.join(dp, f)) for dp,_,fn in os.walk(wt_path) for f in fn) / 1e9
    print(f"[weights] saved {wt_path} ({wt_size:.2f}GB)", flush=True)

    print(f"\n[verdict] {verdict}", flush=True)
    print(f"  M2: {M2_before:.4f} → {M2_after:.4f} ({M2_reduction*100:+.1f}%)", flush=True)
    print(f"  util: {util_before:.3f} → {util_after:.3f} drift={util_after-util_before:+.3f}", flush=True)
    print(f"  entropy: {ent_after:.3f} (max={np.log(32):.3f})", flush=True)
    print(f"  train_loss: {before_train:.4f} → {after_train:.4f}", flush=True)
    print(f"  val_loss:   {before_val:.4f} → {after_val:.4f}", flush=True)
    best_was_loaded = best_val_loss < float('inf')
    if best_was_loaded:
        best_wt_path = os.path.join(BEST_WEIGHTS_DIR, f"granite_pams_align_{OUT_TAG}")
    print(f"  pairwise_cos={avg_pairwise_cos_val:.4f}", flush=True)
    print(f"  flip_B: {np.mean([l['flip_B'] for l in before_B]):.4f} → {flip_B_after:.4f}", flush=True)
    print(f"  best_val_loss: {best_val_loss:.4f} (saved)" if best_was_loaded else "  best_val_loss: N/A (no improvement)", flush=True)
    for r in after_all[:5]:
        c90 = r['cos_buckets'].get('gt0.2',{}).get('cos_median',None)
        print(f"  L{r['layer']}: tear={r['tear_median']:.4f} util={r['expert_util']:.3f} "
              f"ent={r['load_entropy']:.3f} bnd={r['boundary_frac_lt0.05']:.3f}", flush=True)

    best_weights_path = best_wt_path if best_was_loaded else None
    result = {"model": MODEL_ID, "lambda": PAMS_LAMBDA, "tau": tau, "steps": PAMS_STEPS,
        "verdict": verdict, "M2_reduction": M2_reduction,
        "util_delta": util_after - util_before,
        "best_val_loss": best_val_loss if best_was_loaded else None,
        "best_weights_path": best_weights_path,
        "before": {"full": before_all, "flip_B": before_B, "loss": before_train, "val_loss": before_val},
        "after": {"full": after_all, "flip_B": after_B, "flip_A": after_A, "loss": after_train, "val_loss": after_val},
        "checkpoints": checkpoints,
        "weights_path": wt_path}
    with open(OUT_PATH, "w") as f: json.dump(result, f, indent=2)
    print(f"[done] {OUT_PATH}", flush=True)
    for h in handles: h.remove()

if __name__ == "__main__":
    main()
