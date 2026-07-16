"""olmoe_beat_lora.py — OLMoE BEAT 全层 LoRA 安全版

LoRA r=8 on 所有expert FFN, 仅 57M 可训parameter.
显存上限保护 85% (27.8 GB), 不会打崩显示驱动.
"""
import os, sys, json, time, torch, torch.nn.functional as F
import numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_ID = "allenai/OLMoE-1B-7B-0125"
DEVICE = "cuda"
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "datasets--wikitext", "snapshots", "b08601e04326c79dfdd32d625aee71d232d685c3",
    "wikitext-103-raw-v1")

LAMBDA = float(os.environ.get("OLLM_LAMBDA", "1.0"))
STEPS = int(os.environ.get("OLLM_STEPS", "500"))
LR = float(os.environ.get("OLLM_LR", "1e-4"))
WIN = 256
STEP = 128
TAU = 0.02
OUT_TAG = os.environ.get("OLLM_TAG", f"lora_l{str(LAMBDA).replace('.','_')}")
OUT_PATH = os.path.join(OUT_DIR, f"olmoe_beat_lora_{OUT_TAG}.json")
VAL_OFFSET = 600000
TRAIN_HOLDOUT_OFFSET = 500000

# 显存上限保护: 80% + expandable segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.set_per_process_memory_fraction(0.78)
print(f"[mem] limit set to 78% ({0.78 * torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB)", flush=True)

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
    print(f"[data] {len(ids)} tokens", flush=True)
    return ids, tok

def inject_lora(model, r=8):
    """注入 LoRA 适配器到所有expert层, 返回 lora parameter列表"""
    moe_blocks = [(n, m) for n, m in model.named_modules() if isinstance(m, OlmoeSparseMoeBlock)]
    lora_params = []
    for li, (name, blk) in enumerate(moe_blocks):
        E, Dout_gu, Din_gu = blk.experts.gate_up_proj.shape  # [64, 2048, 2048]
        _, Dout_dn, Din_dn = blk.experts.down_proj.shape     # [64, 2048, 1024]
        # gate_up_proj LoRA: W'[2048,2048] = W + A[2048,r]@B[r,2048]
        blk.lora_A_gate = torch.nn.Parameter(torch.randn(E, Dout_gu, r, device=DEVICE, dtype=torch.bfloat16) * 0.01)
        blk.lora_B_gate = torch.nn.Parameter(torch.zeros(E, r, Din_gu, device=DEVICE, dtype=torch.bfloat16))
        # down_proj LoRA: W'[2048,1024] = W + A[2048,r]@B[r,1024]
        blk.lora_A_down = torch.nn.Parameter(torch.randn(E, Dout_dn, r, device=DEVICE, dtype=torch.bfloat16) * 0.01)
        blk.lora_B_down = torch.nn.Parameter(torch.zeros(E, r, Din_dn, device=DEVICE, dtype=torch.bfloat16))
        lora_params.extend([blk.lora_A_gate, blk.lora_B_gate, blk.lora_A_down, blk.lora_B_down])
        # 标记原权重为冻结
        blk.experts.gate_up_proj.requires_grad_(False)
        blk.experts.down_proj.requires_grad_(False)
    total_lora = sum(p.numel() for p in lora_params)
    print(f"[lora] r={r}, {len(moe_blocks)} layers, {total_lora/1e6:.1f}M trainable params", flush=True)
    return lora_params

def align_loss(blocks, tau, k, device):
    total, cnt = 0.0, 0
    for _, blk in blocks:
        if not hasattr(blk, "_last_captured_h"):
            continue
        h = blk._last_captured_h.to(device)
        w = blk.gate.weight.detach().float()
        logits = F.linear(h, w)
        zs = logits.sort(dim=-1, descending=True)
        m = (zs.values[:, k-1] - zs.values[:, k] < tau).cpu()
        if not m.any():
            continue
        ek = zs.indices[:, k-1][m]; ek1 = zs.indices[:, k][m]
        hb = h[m]
        # 缓存 LoRA 适配权重 (每层一次, 不在逐 token 循环内重复计算)
        lora_gate_cache = {}
        lora_down_cache = {}
        gu = blk.experts.gate_up_proj
        dn = blk.experts.down_proj
        for idx in range(m.sum().item()):
            e_i, e_j = int(ek[idx]), int(ek1[idx])
            for e in (e_i, e_j):
                if e not in lora_gate_cache:
                    w_gu = gu[e].float()
                    gu_lora = (blk.lora_A_gate[e].float() @ blk.lora_B_gate[e].float())
                    lora_gate_cache[e] = w_gu + gu_lora
                    w_dn = dn[e].float()
                    dn_lora = (blk.lora_A_down[e].float() @ blk.lora_B_down[e].float())
                    lora_down_cache[e] = w_dn + dn_lora
            hi = hb[idx:idx+1].float()
            gate, up = F.linear(hi, lora_gate_cache[e_i]).chunk(2, dim=-1)
            yi = F.linear(F.silu(gate) * up, lora_down_cache[e_i])
            gate, up = F.linear(hi, lora_gate_cache[e_j]).chunk(2, dim=-1)
            yj = F.linear(F.silu(gate) * up, lora_down_cache[e_j])
            total += (yi-yj).norm()**2 / (yi.norm()**2 + yj.norm()**2 + 1e-8)
            cnt += 1
    return (total/max(cnt,1)).to(device) if cnt > 0 else torch.zeros((), device=device)

def measure_loss(model, ids, n_win=5, device=DEVICE):
    model.eval()
    losses = []
    with torch.no_grad():
        for wi in range(n_win):
            w = ids[wi*STEP:wi*STEP+WIN].unsqueeze(0).to(device)
            o = model(input_ids=w, attention_mask=torch.ones_like(w))
            losses.append(F.cross_entropy(o.logits[0,:-1].float(), w[0,1:]).item())
    model.train()
    if len(losses) == 0:
        return 0.0
    return float(np.mean(losses))

def measure_m2(model, ids, moe_blocks, n_win=3, device=DEVICE):
    """估算 M2 (仅用于verification BEAT 是否起效)"""
    model.eval()
    with torch.no_grad():
        def make_hook_m2(blk):
            def hook(mod, inp, out):
                blk._m2_h = inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu()
                blk._m2_l = out[0].detach().float().cpu()
            return hook
        m2_handles = [b.gate.register_forward_hook(make_hook_m2(b)) for _, b in moe_blocks]
        for wi in range(n_win):
            w = ids[wi*STEP:wi*STEP+WIN].unsqueeze(0).to(device)
            model(input_ids=w, attention_mask=torch.ones_like(w))
        tears = []
        for _, blk in moe_blocks:
            h = blk._m2_h.to(device)
            logits = blk._m2_l.to(device)
            if len(h) == 0: continue
            zs = logits.sort(dim=-1, descending=True)
            ek = zs.indices[:, 7]; ek1 = zs.indices[:, 8]
            yk, yk1 = torch.zeros_like(h), torch.zeros_like(h)
            for e_idx in range(blk.experts.num_experts):
                m_k = (ek == e_idx); m_k1 = (ek1 == e_idx)
                if m_k.any():
                    wg = blk.experts.gate_up_proj[e_idx].float()
                    wd = blk.experts.down_proj[e_idx].float()
                    if hasattr(blk, 'lora_A_gate'):
                        wg = wg + (blk.lora_A_gate[e_idx].float() @ blk.lora_B_gate[e_idx].float())
                        wd = wd + (blk.lora_A_down[e_idx].float() @ blk.lora_B_down[e_idx].float())
                    g, u = F.linear(h[m_k], wg).chunk(2, dim=-1)
                    yk[m_k] = F.linear(F.silu(g) * u, wd)
                    if (ek1 == e_idx).any():
                        g1, u1 = F.linear(h[ek1 == e_idx], wg).chunk(2, dim=-1)
                        yk1[ek1 == e_idx] = F.linear(F.silu(g1) * u1, wd)
            n = (yk - yk1).norm(dim=-1)
            d = yk.norm(dim=-1) + yk1.norm(dim=-1) + 1e-6
            tears.append(float((n/d).median().cpu().numpy()))
        for h in m2_handles: h.remove()
    model.train()
    return float(np.mean(tears)) if tears else 0.0

def main():
    torch.manual_seed(42)
    print(f"[olmoe_beat_lora] λ={LAMBDA} steps={STEPS} lr={LR}", flush=True)

    ids, _ = load_data()
    ids_val = ids[VAL_OFFSET:VAL_OFFSET+WIN*5]
    ids_tr = ids[TRAIN_HOLDOUT_OFFSET:TRAIN_HOLDOUT_OFFSET+WIN*5]

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model.train()
    model.gradient_checkpointing_enable()

    for p in model.parameters():
        p.requires_grad_(False)

    moe_blocks = [(n, m) for n, m in model.named_modules() if isinstance(m, OlmoeSparseMoeBlock)]
    lora_params = inject_lora(model, r=8)

    # Hook: 存储 hidden states (detached)
    def make_hook(blk):
        def hook(mod, inp, out):
            blk._last_captured_h = inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu()
        return hook
    handles = [b.gate.register_forward_hook(make_hook(b)) for _, b in moe_blocks]

    # measurement前
    before_loss = measure_loss(model, ids_tr, n_win=3)
    before_val = measure_loss(model, ids_val, n_win=3)
    before_m2 = measure_m2(model, ids_val, moe_blocks, n_win=3)
    print(f"[before] train={before_loss:.4f} val={before_val:.4f} M2={before_m2:.4f}", flush=True)

    # 训练
    opt = torch.optim.AdamW(lora_params, lr=LR, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    loss_log = []; t0 = time.time()

    for step in range(STEPS):
        start = (step * STEP) % (len(ids) - WIN)
        wid = ids[start:start+WIN].unsqueeze(0).to(DEVICE)
        out = model(**dict(input_ids=wid, attention_mask=torch.ones_like(wid)))
        ce = F.cross_entropy(out.logits[0, :-1].float(), wid[0, 1:])
        align = align_loss(moe_blocks, TAU, 8, DEVICE)
        total = ce + LAMBDA * align
        total.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step(); opt.zero_grad(); sch.step()
        loss_log.append(float(ce.detach()))
        if (step+1) % 100 == 0:
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  [{step+1}/{STEPS}] ce={np.mean(loss_log[-50:]):.4f} align={align:.4f} mem={mem:.1f}GB {time.time()-t0:.0f}s", flush=True)

    # measurement后
    after_loss = measure_loss(model, ids_tr, n_win=3)
    after_val = measure_loss(model, ids_val, n_win=3)
    after_m2 = measure_m2(model, ids_val, moe_blocks, n_win=3)
    print(f"[after] train={after_loss:.4f} val={after_val:.4f} M2={after_m2:.4f}", flush=True)

    # saving LoRA 权重
    lora_state = {k: v.clone().cpu() for k, v in zip(
        ["lora_A_gate", "lora_B_gate", "lora_A_down", "lora_B_down"],
        [blk.lora_A_gate.data.cpu() for _, blk in moe_blocks[:1]]  # wrong, need per layer
    )}
    # Actually save per-layer
    lora_sd = {}
    for li, (name, blk) in enumerate(moe_blocks):
        lora_sd[f"L{li}_A_gate"] = blk.lora_A_gate.data.cpu()
        lora_sd[f"L{li}_B_gate"] = blk.lora_B_gate.data.cpu()
        lora_sd[f"L{li}_A_down"] = blk.lora_A_down.data.cpu()
        lora_sd[f"L{li}_B_down"] = blk.lora_B_down.data.cpu()
    lora_path = OUT_PATH.replace(".json", "_lora.pt")
    torch.save(lora_sd, lora_path)
    lora_mb = sum(v.numel() for v in lora_sd.values()) * 4 / 1e6
    print(f"[lora] saved {lora_path} ({lora_mb:.0f}MB)", flush=True)

    result = {"model": MODEL_ID, "lambda": LAMBDA, "steps": STEPS, "lora_r": 8,
        "trainable_M": sum(p.numel() for p in lora_params) / 1e6,
        "before": {"loss": before_loss, "val_loss": before_val, "M2": before_m2},
        "after": {"loss": after_loss, "val_loss": after_val, "M2": after_m2},
        "ce_trace": loss_log, "lora_weights": lora_path}
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] {OUT_PATH}", flush=True)
    for h in handles:
        h.remove()

if __name__ == "__main__":
    main()
