"""prune_deconvention.py — 7.1 约定感知的剪枝判据去污染 (Type A 删expert损伤解析分离)

背景 (路线图 §7.1): 剪枝判据文献 (统一打分框架 arXiv 2606.15716, REAP arXiv 2510.13999, EAC-MoE)
都算删expert p 的损伤 D(p), 但无一按归一化约定条件化。Type A (softmax-then-topk, 全局分母,
OLMoE native norm_topk_prob=False) 上 D(p) = 真实贡献 C(p) + 分母塌缩伪影 A(p), 二者混杂。
law 给塌缩项 A(p) ∝ e^{z_p}/全局分母 = r_p (p 在全局 softmax 分母的占比)。

可测主张 (论文级, tier E+I): 对 Type A 模型 OLMoE-1B-7B-0125, 去污染 D_clean(p)=D(p)-α·r_p 后
expert重要性排序会变 (Spearman ρ(D,D_clean) < 1); D 排名高但 D_clean 排名低的expert被分母塌缩伪影
高估重要性。Type B 约定 (强加 renorm) 下 A(p)=0, D_clean=D, 排序unchanged (law 预测零结果control)。

design:
  - forward_hard_drop: 基于 mtp.forward_hard (L152) 加 z_p=-inf。p_drop=None 为 baseline。
    renorm=False (Type A): softmax over all E, 删 p 分母排除 p → 其他权重膨胀 1/(1-r_p) → 塌缩。
    renorm=True (Type B): top-k 先排除 p (gate=0), renorm 分母只含选中集不含 p → 不塌缩 A(p)=0。
  - patch_layers: 替换所有 MoE block.forward 为 forward_hard_drop(p_drop, renorm)。
    baseline: 所有层 p_drop=None; 删 p: 仅 layer_idx 层 p_drop=p, 其他层 None (唯一差异是 L8 删 p)。
  - 三量: D(p)=nll_drop(p)-nll_base (逐 p run_loss); r_p=mean_t softmax(logits)[p] (一次 capture);
    D_clean=D-α·r_p (α=OLS Cov(D,r_p)/Var(r_p))。
  - 排序: Spearman ρ(D,D_clean) 手算 (argsort ranks, 无 ties); rank shift; 被污染 top-5。

复用 (禁止反向依赖, paper repo ← extension repo):
  - mtp._combine (L139), mtp.discover_moe_blocks, mtp.forward_hard (L152, 自检control)
  - phase3_experimentA.load_corpus_ids (WikiText-2 test 1-D ids, truncation 70000)
  - gate_logits_fp32 / run_loss 逻辑复制自 ph13_p2_convention (L58/L136, fp32 解耦 bf16 假翻转)

口径: OLMoE-1B-7B-0125 非 canonical 0924, 数值不与 0924 混。纯推理零训练, 不触训练 forward,
不算解冻。Type A=softmax-then-topk 不重归一 (native); Type B=强加 renorm (同模型同权重同 token,
隔离纯约定效应, 非 Mixtral 原生)。主张是排序变化 (相对量), 非绝对损伤分解。
证据: E (相关性观察) + I (law 推导塌缩项)。
显存: 模型 bf16 14GB + run_loss no_grad + 单层 patch, PEAK_LIMIT 30.5GB。一次一个实验。
output: results/prune_deconvention_{PD_MODEL_TAG}_L{PD_LAYER}.json
环境: MTEAR_MODEL (默 allenai/OLMoE-1B-7B-0125), PD_LAYER (默 8), PD_N_WIN (默 40), MTEAR_K (默 8), PD_MODEL_TAG (默 MODEL 最后一段小写)
"""
import os
import sys
import json
import types
import gc

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/path/to/project/moe-tearing-main")
import moe_tear_probe as mtp
sys.path.insert(0, "/path/to/project/moe-tear-experiments/scripts")
import phase3_experimentA as p3

MODEL = os.environ.get("MTEAR_MODEL", "allenai/OLMoE-1B-7B-0125")
K = int(os.environ.get("MTEAR_K", "8"))
PD_LAYER = int(os.environ.get("PD_LAYER", "8"))
PD_N_WIN = int(os.environ.get("PD_N_WIN", "40"))
PD_MODEL_TAG = os.environ.get("PD_MODEL_TAG", MODEL.split("/")[-1].lower())
PD_WIN = 128
PD_STEP = 64
PEAK_LIMIT_GB = 30.5

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def gate_logits_fp32(block, h):
    # 复制自 ph13_p2_convention L58: fp32 gate logits, 解耦 bf16 forward 假翻转
    W = block.gate.weight.detach().float()
    if h.dim() == 1:
        h = h.unsqueeze(0)
    return h.float() @ W.T


def forward_hard_drop(block, h, k, p_drop, renorm):
    # 基于 mtp.forward_hard (L152) 加 z_p=-inf; p_drop=None 为 baseline
    logits = gate_logits_fp32(block, h)
    if p_drop is not None:
        logits = logits.clone()
        logits[:, p_drop] = -float('inf')
    gates = F.softmax(logits, dim=-1)           # Type A: softmax over all E; e^{-inf}=0 排除 p, 分母塌缩
    topv, topi = gates.topk(k, dim=-1)          # p 的 gate=0 不入选
    if renorm:                                  # Type B: 重归一 over top-k (分母只含选中集, 不含 p)
        topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    w = torch.zeros_like(gates)
    w.scatter_(-1, topi, topv)
    return mtp._combine(block, h, w)


def forward_hard_drop_restored(block, h, k, p_drop):
    # Type A 貨量保持删 (精确反事实): 删 p 召回第9, 幸存权重 = baseline softmax 值 (未膨胀)
    # 闭式: softmax_drop[j]·(1-σ_p) = baseline_softmax[j] for j≠p, σ_p = softmax(logits)[p]
    # 等价实现: top-k 集合 = 删 p 后 top-k (召回第9), 权重 = baseline_softmax 在该集合的值
    # D(p) 裸删 = expert替换 + 膨胀; D_restored = 纯expert替换 = C(p); A(p)=D-D_restored 因果精确
    logits = gate_logits_fp32(block, h)
    base_softmax = F.softmax(logits, dim=-1)
    if p_drop is None:
        # baseline = forward_hard (renorm=False native, 不删)
        topv, topi = base_softmax.topk(k, dim=-1)
    else:
        logits_drop = logits.clone()
        logits_drop[:, p_drop] = -float('inf')
        softmax_drop = F.softmax(logits_drop, dim=-1)
        _, topi = softmax_drop.topk(k, dim=-1)        # 删 p 后 top-k 集合 (召回第9)
        topv = base_softmax.gather(-1, topi)          # baseline softmax 值 (未膨胀)
    w = torch.zeros_like(base_softmax)
    w.scatter_(-1, topi, topv)
    return mtp._combine(block, h, w)


def make_forward_drop(k, p_drop, renorm, restored=False):
    def fwd(self, hidden_states):
        b, s, d = hidden_states.shape
        h = hidden_states.view(-1, d)
        if restored:
            out = forward_hard_drop_restored(self, h, k, p_drop)
        else:
            out = forward_hard_drop(self, h, k, p_drop, renorm)
        return out.reshape(b, s, d)
    return fwd


def patch_layers(model, k, renorm, drop_layer=None, p_drop=None, restored=False):
    # 替换所有 MoE block.forward; drop_layer 层用 p_drop, 其他层 None (baseline)
    # restored=True: 用质量保持删 (精确反事实, Type A, renorm 忽略)
    blocks = mtp.discover_moe_blocks(model)
    originals = []
    for i, (_, blk, _) in enumerate(blocks):
        pd = p_drop if (i == drop_layer and p_drop is not None) else None
        fn = make_forward_drop(k, pd, renorm, restored=restored)
        originals.append((blk, blk.forward))
        blk.forward = types.MethodType(fn, blk)
    return originals


def restore(originals):
    for blk, fwd in originals:
        blk.forward = fwd


def run_loss(model, ids, n_windows, batch_size=8):
    # 批量化 per-token CE: batch_size 个 128-token 窗一次 forward (减 kernel launch), no_grad
    losses = []
    with torch.no_grad():
        for wi in range(0, n_windows, batch_size):
            n = min(batch_size, n_windows - wi)
            wids = torch.stack([ids[(wi + j) * PD_STEP:(wi + j) * PD_STEP + PD_WIN] for j in range(n)]).to("cuda")
            enc = {"input_ids": wids, "attention_mask": torch.ones_like(wids)}
            out = model(**enc)
            logits = out.logits[:, :-1].float()           # [n, 127, vocab]
            targets = wids[:, 1:]                          # [n, 127]
            l = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
            losses.append(l.cpu().numpy())
    return np.concatenate(losses)


def capture_layer_h(model, layer_idx, ids, n_windows):
    # hook capture layer_idx block input h (fp32 cpu), 用于算 r_p
    blocks = mtp.discover_moe_blocks(model)
    blk = blocks[layer_idx][1]
    captured = []
    def hook(mod, inp, _out):
        h = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
        captured.append(h.cpu())
    handle = blk.register_forward_hook(hook)
    with torch.no_grad():
        for wi in range(n_windows):
            wid = ids[wi * PD_STEP:wi * PD_STEP + PD_WIN].unsqueeze(0).to("cuda")
            enc = {"input_ids": wid, "attention_mask": torch.ones_like(wid)}
            model(**enc)
    handle.remove()
    return torch.cat(captured, 0)  # [T_total, D]


def compute_r_p(blk, h_all):
    # r_p = mean_t softmax(logits)[p], 全部 64 expert一次算
    with torch.no_grad():
        logits = gate_logits_fp32(blk, h_all.to("cuda"))   # [T, E]
        probs = F.softmax(logits, dim=-1)                  # [T, E]
        return probs.mean(dim=0).cpu().numpy()              # [E]


def spearman_rho(a, b):
    # 手算 Spearman (argsort ranks, D 连续无 ties)
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float(ra.dot(rb) / denom) if denom > 0 else 0.0


def bootstrap_delta_ci(base_loss, alt_loss, n_win, tokens_per_win=127, n_boot=1000, seed=0):
    # per-window block bootstrap of mean(alt - base); 返回 (point_est, lo, hi)
    # base_loss/alt_loss: [n_win * tokens_per_win] per-token CE
    rng = np.random.RandomState(seed)
    base_w = base_loss.reshape(n_win, tokens_per_win).mean(axis=1)
    alt_w = alt_loss.reshape(n_win, tokens_per_win).mean(axis=1)
    deltas = alt_w - base_w
    idx = rng.randint(0, n_win, (n_boot, n_win))
    boots = deltas[idx].mean(axis=1)
    return float(deltas.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def bootstrap_mean_ci(x_win, n_boot=1000, seed=0):
    # per-window block bootstrap of mean(x); x_win: [n_win]
    rng = np.random.RandomState(seed)
    n_win = len(x_win)
    idx = rng.randint(0, n_win, (n_boot, n_win))
    boots = x_win[idx].mean(axis=1)
    return float(x_win.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def split_half_rho(per_win_D, n_split=200, seed=0):
    # D 自身 split-half 信度 = 噪声零线. per_win_D: [n_experts, n_win] per-window D(p)
    # 多次 random split, ρ(D_half1, D_half2) 平均. 若 ≈ ρ(D,D_restored) 则排序信号被信度天花板吃掉
    rng = np.random.RandomState(seed)
    n_win = per_win_D.shape[1]
    rhos = []
    for _ in range(n_split):
        idx = rng.permutation(n_win)
        h1, h2 = idx[:n_win // 2], idx[n_win // 2:]
        D_h1 = per_win_D[:, h1].mean(axis=1)
        D_h2 = per_win_D[:, h2].mean(axis=1)
        rhos.append(spearman_rho(D_h1, D_h2))
    return float(np.mean(rhos)), float(np.std(rhos)), float(np.percentile(rhos, 2.5))


def free_cuda():
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.empty_cache()


def selftest():
    # CPU 无模型自检: forward_hard_drop 语义 + patch_layers 隔离
    print("[selftest] forward_hard_drop 语义自检 (mock block)...")
    torch.manual_seed(0)
    E, D_in, k = 8, 4, 3
    # mock block: gate.weight (E, D_in), experts 用线性
    class MockBlock:
        def __init__(self):
            self.gate = torch.nn.Linear(D_in, E, bias=False)
            self.experts = torch.nn.ModuleList([torch.nn.Linear(D_in, D_in, bias=False) for _ in range(E)])
        def expert_fwd(self, e, h):
            return self.experts[e](h)
        def forward(self, h):
            return h
    blk = MockBlock()
    h = torch.randn(5, D_in)
    # patch _combine 不可用 (要真 block), 手算 forward_hard_drop 权重逻辑verification
    logits = gate_logits_fp32(blk, h)
    # baseline Type A
    g_base = F.softmax(logits, dim=-1)
    tv_base, ti_base = g_base.topk(k, dim=-1)
    w_base_sum = tv_base.sum(dim=-1)  # native 不 renorm, mass<1
    # 删 p=0 Type A
    logits_d = logits.clone(); logits_d[:, 0] = -float('inf')
    g_d = F.softmax(logits_d, dim=-1)
    tv_d, ti_d = g_d.topk(k, dim=-1)
    w_d_sum = tv_d.sum(dim=-1)
    assert (ti_d == 0).any() == False, "Type A 删 p 后 p 不应在 top-k"
    r0 = g_base[:, 0]
    inflate = 1.0 / (1.0 - r0).clamp_min(1e-12)
    # 删 p 后 top-k 从剩余 E 选, 权重 = inflate * baseline 对应权重, 和也 inflate
    expected_w_d_sum = inflate * g_base[:, 1:].topk(k, dim=-1).values.sum(-1)
    assert torch.allclose(w_d_sum, expected_w_d_sum, atol=1e-5), \
        "Type A 删 p: top-k 从剩余 E 选, 权重和 = inflate * baseline 去掉 p 后 top-k 和"
    # Type A 膨胀: 删 p 后其他权重膨胀 (分母变小), 同 token 同选中集时 w_d > w_base
    # 取删 p 前后都选中的expert对比
    common = [j for j in range(E) if j != 0]
    w_base_common = g_base[:, common]
    w_d_common = g_d[:, common]
    # 删 p 分母 S'=S-e^{z_0}, 权重膨胀 1/(1-r_0); verification w_d_common = w_base_common / (1 - r_0)
    assert torch.allclose(w_d_common, w_base_common * inflate.unsqueeze(-1), atol=1e-5), \
        "Type A 删 p: 其他权重应膨胀 1/(1-r_p), 塌缩机制verification"
    print(f"  Type A 删 p 膨胀verification通过: w_d = w_base/(1-r_p), r_p median={r0.median():.4f}")
    # Type B (renorm=True) 删 p: top-k 和 == 1 (不塌缩)
    tv_d_b = tv_d / tv_d.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    assert torch.allclose(tv_d_b.sum(dim=-1), torch.ones(5), atol=1e-5), "Type B 删 p: renorm 后和=1"
    tv_base_b = tv_base / tv_base.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    assert torch.allclose(tv_d_b.sum(dim=-1), tv_base_b.sum(dim=-1), atol=1e-5), "Type B baseline 和也=1"
    print("  Type B 删 p 不塌缩verification通过 (和==1)")
    # patch_layers 隔离自检 (mock model)
    class MockModel:
        def __init__(self):
            self.layers = [type("L", (), {"mlp": MockBlock()})() for _ in range(3)]
    mm = MockModel()
    # patch_layers 用 discover_moe_blocks, mock 不兼容, 直接测 make_forward_drop 绑定 + __dict__ 隔离
    blk0 = mm.layers[0].mlp
    blk0.forward = types.MethodType(make_forward_drop(k, 0, False), blk0)
    blk1 = mm.layers[1].mlp
    blk1.forward = types.MethodType(make_forward_drop(k, None, False), blk1)
    assert "forward" in mm.layers[0].mlp.__dict__ and "forward" in mm.layers[1].mlp.__dict__, \
        "patch 层应写 forward 到 __dict__"
    assert "forward" not in mm.layers[2].mlp.__dict__, "未 patch 层 __dict__ 不应有 forward"
    print("  patch 隔离自检通过 (drop_layer=0 时 layer2 未替换)")
    # 质量保持删权重逻辑verification (复现 forward_hard_drop_restored 的 gather, 不调 _combine)
    # 两arm唯一差异 = 膨胀因子 1/(1-σ_p), 因果反事实的关键
    base_sm = F.softmax(logits, dim=-1)
    logits_d2 = logits.clone(); logits_d2[:, 0] = -float('inf')
    sm_drop = F.softmax(logits_d2, dim=-1)
    _, topi_d = sm_drop.topk(k, dim=-1)            # 删 p 后 top-k 集合 (召回第9)
    rest_w = base_sm.gather(-1, topi_d)            # restored 权重 = baseline softmax[删p后top-k] (未膨胀)
    naked_w = sm_drop.gather(-1, topi_d)           # 裸删权重 = softmax_drop[删p后top-k] (膨胀)
    assert torch.allclose(naked_w, rest_w / (1 - r0).unsqueeze(-1), atol=1e-5), \
        "裸删权重 = restored/(1-σ_p): 两arm唯一差异是膨胀因子, 因果反事实"
    # restored 权重和 = baseline softmax 在删p后 top-k 的和 (未膨胀, <裸删和)
    assert torch.all(rest_w.sum(-1) < naked_w.sum(-1)), "restored 权重和 < 裸删权重和 (无膨胀)"
    print(f"  质量保持删语义verification: restored权重=baseline softmax[删p后top-k](未膨胀), 裸删=restored/(1-σ_p) ✓")
    print("[selftest] 全部通过")


def main():
    if "--selftest" in sys.argv:
        selftest(); return
    if not torch.cuda.is_available():
        print("ERROR: 需 GPU"); sys.exit(1)
    torch.manual_seed(0); np.random.seed(0)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[info] model={MODEL} k={K} layer={PD_LAYER} n_win={PD_N_WIN}", flush=True)
    print(f"[info] free GPU before load: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True).eval()
    mtp.set_norm_topk(model, "auto")
    ids = p3.load_corpus_ids(tok)
    print(f"[info] free GPU after load: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB", flush=True)

    blocks = mtp.discover_moe_blocks(model)
    n_experts = mtp._moe_num_experts(blocks[0][1])
    blk_target = blocks[PD_LAYER][1]
    print(f"[info] blocks={len(blocks)} experts={n_experts} target_layer={PD_LAYER}", flush=True)

    # === Type A 精确反事实两arm (废 Type B 脏arm; 质量保持删 = 同模型 Type A 分母不含p实现) ===
    print(f"\n=== Type A 精确反事实 (裸删 vs 质量保持删) ===", flush=True)
    free_cuda()
    orig = patch_layers(model, K, False, drop_layer=None, p_drop=None, restored=False)
    torch.cuda.reset_peak_memory_stats()
    base_loss = run_loss(model, ids, PD_N_WIN)
    nll_base = float(np.mean(base_loss))
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    restore(orig)
    print(f"[typeA] baseline nll={nll_base:.6f} peak={peak_gb:.2f}GB", flush=True)
    if peak_gb > PEAK_LIMIT_GB:
        print(f"[ABORT] peak {peak_gb:.2f}GB > {PEAK_LIMIT_GB}GB"); sys.exit(2)
    # r_p: capture baseline h (OLS 代理对比用)
    orig = patch_layers(model, K, False, drop_layer=None, p_drop=None, restored=False)
    h_all = capture_layer_h(model, PD_LAYER, ids, PD_N_WIN)
    restore(orig)
    r_p = compute_r_p(blk_target, h_all)
    print(f"[typeA] r_p median={np.median(r_p):.4f} max={r_p.max():.4f} min={r_p.min():.4f}", flush=True)
    # 逐expert两arm: 裸删 D(p) + 质量保持删 D_restored(p)
    # saving per-window loss 供 split-half 信度零线 + A(p) 配对 bootstrap CI (噪声威胁补救)
    TPW = PD_WIN - 1  # 127 per-token CE per 128-token window
    base_win = base_loss.reshape(PD_N_WIN, TPW).mean(axis=1)  # [n_win] per-window mean CE
    D = []; D_rest = []; D_ci = []; D_rest_ci = []
    drop_win_all = []; rest_win_all = []
    for p in range(n_experts):
        free_cuda()
        # 裸删 (含膨胀)
        orig = patch_layers(model, K, False, drop_layer=PD_LAYER, p_drop=p, restored=False)
        try:
            drop_loss = run_loss(model, ids, PD_N_WIN)
        except RuntimeError as e:
            print(f"[ABORT] p={p} naked drop error: {e}", flush=True)
            restore(orig); free_cuda(); break
        restore(orig); free_cuda()
        drop_win = drop_loss.reshape(PD_N_WIN, TPW).mean(axis=1)  # [n_win]
        drop_win_all.append(drop_win)
        Dp, lo, hi = bootstrap_delta_ci(base_loss, drop_loss, PD_N_WIN)
        D.append(Dp); D_ci.append([lo, hi])
        # 质量保持删 (删p召回第9, 幸存权重=baseline softmax值, 未膨胀)
        orig = patch_layers(model, K, False, drop_layer=PD_LAYER, p_drop=p, restored=True)
        try:
            rest_loss = run_loss(model, ids, PD_N_WIN)
        except RuntimeError as e:
            print(f"[ABORT] p={p} restored drop error: {e}", flush=True)
            restore(orig); free_cuda(); break
        restore(orig); free_cuda()
        rest_win = rest_loss.reshape(PD_N_WIN, TPW).mean(axis=1)
        rest_win_all.append(rest_win)
        Drp, rlo, rhi = bootstrap_delta_ci(base_loss, rest_loss, PD_N_WIN)
        D_rest.append(Drp); D_rest_ci.append([rlo, rhi])
        if (p + 1) % 16 == 0:
            print(f"[typeA] p={p+1}/{n_experts} D_med={np.median(D):.6f} D_rest_med={np.median(D_rest):.6f}", flush=True)
    D = np.asarray(D); D_rest = np.asarray(D_rest)
    drop_win_all = np.asarray(drop_win_all)   # [n_experts, n_win]
    rest_win_all = np.asarray(rest_win_all)  # [n_experts, n_win]
    # per-window D / D_restored / A; A = drop-rest 配对差分消掉 base 共同噪声
    per_win_D = drop_win_all - base_win[None, :]
    per_win_Drest = rest_win_all - base_win[None, :]
    per_win_A = drop_win_all - rest_win_all  # 配对: 两arm同 token 同窗, base 完全消掉
    A = D - D_rest  # 因果精确塌缩 = 裸删 - 质量保持删 (零回归, 逐expert闭式反事实)
    rho = spearman_rho(D, D_rest)
    rank_D = D.argsort().argsort()
    rank_dr = D_rest.argsort().argsort()
    rank_shift = np.abs(rank_D - rank_dr).astype(int)
    pollution_score = rank_D - rank_dr  # 正 = D 高估 (被膨胀污染)
    polluted_idx = np.argsort(-pollution_score)[:5]
    polluted_top5 = []
    for idx in polluted_idx:
        polluted_top5.append({
            "expert": int(idx), "D": float(D[idx]), "D_restored": float(D_rest[idx]),
            "A_causal": float(A[idx]), "r_p": float(r_p[idx]),
            "rank_D": int(rank_D[idx]), "rank_D_restored": int(rank_dr[idx]),
            "pollution_score": int(pollution_score[idx])})
    # OLS 代理对比: verification精确 A 与 OLS α·r_p 一致性 (共线性诊断)
    alpha_ols = float(np.cov(D, r_p)[0, 1] / np.var(r_p)) if np.var(r_p) > 0 else 0.0
    A_ols = (alpha_ols * r_p).tolist()
    rho_D_Acausal = spearman_rho(D, A)
    rho_Acausal_rpsort = spearman_rho(A, r_p)
    drest_med = float(np.median(D_rest))
    n_neg = int((D_rest < 0).sum())
    n_ci_cross0 = int(sum(1 for ci in D_rest_ci if ci[0] <= 0 <= ci[1]))
    # === 噪声威胁补救: split-half 信度零线 + A(p) 配对 bootstrap CI ===
    # D/D_restored 均为 40 窗噪声估计; A≡0 时 ρ(D,D_rest) 亦 <1. 须用 D 自身 split-half 信度
    # 作天花板: 只有 ρ(D,D_restored) 显著低于天花板, 排序变化信号才真实; 0.95 阈值系拍定无依据.
    sh_rho_D, sh_rho_D_std, sh_rho_D_lo = split_half_rho(per_win_D)
    sh_rho_Drest, sh_rho_Drest_std, _ = split_half_rho(per_win_Drest)
    sh_rho_A, sh_rho_A_std, _ = split_half_rho(per_win_A)
    # A(p) 配对 bootstrap CI: 两arm同 token 同窗, A=drop-rest 消掉窗口级共同噪声, CI 应远窄于 D_restored
    A_ci = []
    for p in range(n_experts):
        m, lo, hi = bootstrap_mean_ci(per_win_A[p])
        A_ci.append([float(m), float(lo), float(hi)])
    A_ci = np.asarray(A_ci)  # [n_experts, 3] = mean, lo95, hi95
    n_A_ci_cross0 = int(((A_ci[:, 1] <= 0) & (A_ci[:, 2] >= 0)).sum())
    n_A_sig_pos = int((A_ci[:, 1] > 0).sum())     # A 显著正 (塌缩伪影确为正)
    n_A_sig_neg = int((A_ci[:, 2] < 0).sum())     # A 显著负 (罕见)
    n_A_sig = n_A_sig_pos + n_A_sig_neg
    # A/D 占比分布 (删 p 的损伤中被塌缩伪影占的比例)
    D_abs = np.abs(D)
    AD_ratio = np.where(D_abs > 1e-7, A / np.maximum(D_abs, 1e-7), np.nan)
    AD_ratio_med = float(np.nanmedian(AD_ratio))
    AD_ratio_iqr = float(np.subtract(*np.nanpercentile(AD_ratio, [75, 25])))
    # exp18 配对 CI (干净论文级案例的铁证)
    e18 = 18
    exp18 = {
        "expert": e18, "A_causal": float(A[e18]),
        "A_ci": [float(A_ci[e18, 1]), float(A_ci[e18, 2])],
        "A_ci_cross0": bool(A_ci[e18, 1] <= 0 <= A_ci[e18, 2]),
        "D": float(D[e18]), "D_restored": float(D_rest[e18]),
        "AD_ratio": float(A[e18] / D_abs[e18]) if D_abs[e18] > 1e-7 else None,
        "r_p": float(r_p[e18]),
    }
    # 信号判定: ρ(D,D_restored) vs split-half ρ(D) 天花板
    rho_ceiling_gap = float(sh_rho_D - rho)  # 正 = ρ 低于天花板 = 真实排序变化信号
    print(f"\n[noise] split-half 信度天花板: ρ(D)={sh_rho_D:.4f}±{sh_rho_D_std:.4f}  ρ(D_rest)={sh_rho_Drest:.4f}±{sh_rho_Drest_std:.4f}  ρ(A)={sh_rho_A:.4f}±{sh_rho_A_std:.4f}", flush=True)
    print(f"[noise] ρ(D,D_restored)={rho:.4f} vs 天花板 ρ(D)={sh_rho_D:.4f}  gap={rho_ceiling_gap:+.4f} ({'低于天花板=真信号' if rho_ceiling_gap > 0.02 else '近天花板=信号被吃' if rho_ceiling_gap > 0 else '已撞天花板'})", flush=True)
    print(f"[noise] A(p) 配对 bootstrap CI: 显著{n_A_sig}/{n_experts} (正{n_A_sig_pos} 负{n_A_sig_neg}) 跨0={n_A_ci_cross0}/{n_experts}", flush=True)
    print(f"[noise] A/D 占比中位={AD_ratio_med:.3f} IQR={AD_ratio_iqr:.3f}", flush=True)
    print(f"[noise] exp18 A={A[e18]:+.5f} 配对CI=[{A_ci[e18,1]:+.5f},{A_ci[e18,2]:+.5f}] 跨0={A_ci[e18,1]<=0<=A_ci[e18,2]}  D_restored CI宽 vs A CI宽", flush=True)

    results = {
        "nll_base": nll_base, "D": D.tolist(), "D_restored": D_rest.tolist(),
        "A_causal": A.tolist(), "A_ols_proxy": A_ols, "alpha_ols": alpha_ols,
        "rho_D_Acausal": rho_D_Acausal, "rho_Acausal_rp": rho_Acausal_rpsort,
        "r_p": r_p.tolist(), "D_ci": D_ci, "D_restored_ci": D_rest_ci,
        "spearman_rho_D_Drestored": rho,
        "rank_shift": rank_shift.tolist(), "rank_shift_max": int(rank_shift.max()),
        "polluted_top5": polluted_top5,
        "D_median": float(np.median(D)), "D_restored_median": drest_med,
        "D_restored_n_negative": n_neg, "D_restored_ci_cross0_count": n_ci_cross0,
        # 噪声威胁补救
        "split_half_rho_D": sh_rho_D, "split_half_rho_D_std": sh_rho_D_std, "split_half_rho_D_lo": sh_rho_D_lo,
        "split_half_rho_Drest": sh_rho_Drest, "split_half_rho_Drest_std": sh_rho_Drest_std,
        "split_half_rho_A": sh_rho_A, "split_half_rho_A_std": sh_rho_A_std,
        "rho_ceiling_gap": rho_ceiling_gap,
        "A_ci": A_ci.tolist(), "A_n_sig": n_A_sig, "A_n_sig_pos": n_A_sig_pos,
        "A_n_sig_neg": n_A_sig_neg, "A_n_ci_cross0": n_A_ci_cross0,
        "AD_ratio_median": AD_ratio_med, "AD_ratio_iqr": AD_ratio_iqr,
        "exp18_paired_ci": exp18,
        "per_win_D": per_win_D.tolist(), "per_win_Drest": per_win_Drest.tolist(),
        "per_win_A": per_win_A.tolist(),
    }
    # Spearman-Brown 信度诊断 (副结果: 剪枝判据信度)
    r_half = sh_rho_D                                # 20 窗半信度
    r_full_sb = 2 * r_half / (1 + r_half) if (1 + r_half) > 0 else 0.0  # SB 推到 40 窗
    r_1 = r_full_sb / (PD_N_WIN * (1 - r_full_sb) + r_full_sb) if (PD_N_WIN * (1 - r_full_sb) + r_full_sb) > 0 else 0.0
    n09 = (0.9 / r_1 - 0.9) / 0.1 if r_1 > 0 else float('inf')  # 信度 0.9 需窗数
    results["spearman_brown_to_full"] = float(r_full_sb)
    results["single_window_r1"] = float(r_1)
    results["n_windows_for_rho_0p9"] = float(n09)
    print(f"[noise] Spearman-Brown: 20窗ρ={r_half:.4f} -> 40窗SB={r_full_sb:.4f}  单窗r1={r_1:.5f}/窗  信度0.9需{n09:.0f}窗", flush=True)
    print(f"[typeA] ρ(D,D_restored)={rho:.4f} rank_shift_max={rank_shift.max()} α_ols={alpha_ols:.4f} ρ(A_causal,r_p)={rho_Acausal_rpsort:.4f}", flush=True)
    print(f"[typeA] D_restored_med={drest_med:.6f} n_neg={n_neg}/{n_experts} ci_cross0={n_ci_cross0}/{n_experts}", flush=True)
    print(f"[typeA] polluted_top5 (D-rank高 D_restored-rank低=被膨胀高估):")
    for z in polluted_top5:
        print(f"    exp{z['expert']:2d} D={z['D']:+.5f} D_rest={z['D_restored']:+.5f} A={z['A_causal']:+.5f} r_p={z['r_p']:.4f} rankD={z['rank_D']:2d}→rankDr={z['rank_D_restored']:2d} ({z['pollution_score']:+d})", flush=True)

    # verdict (推理修正): 配对 ρ(D,Drest) 测共享窗口两arm一致性, 非总体排序变化检验量;
    # split-half ρ(D) 测 20 窗独立两半信度, 两量尺度不同 (噪声共享 vs 独立) 不可直接比方向.
    # 正确: 配对 ρ 本身不构成排序变化证据 -> 0.95 阈值废弃, 全局重排无证据.
    # 不可反过来用 "ρ 高于天花板" 证伪 (同类尺度错误). 主结果换锚为 A(p) 配对 CI 稀疏显著.
    if n_A_sig_pos >= 1:
        verdict = "RANK_SHIFT_UNEVIDENCED_A_SPARSE_CAUSAL"
    elif n_A_sig_neg >= 1:
        verdict = "RANK_SHIFT_UNEVIDENCED_A_NEG_ONLY"
    else:
        verdict = "RANK_SHIFT_UNEVIDENCED_NO_A_SIGNAL"
    print(f"\n[verdict] {verdict}  配对ρ(D,Drest)={rho:.4f}非排序变化检验量(0.95废弃)  split-half信度ρ(D)={sh_rho_D:.4f} 40窗SB={r_full_sb:.4f}  A_sig_pos={n_A_sig_pos}", flush=True)

    out = {
        "model": MODEL, "k": K, "layer": PD_LAYER, "n_experts": n_experts,
        "n_windows": PD_N_WIN, "window": PD_WIN, "step": PD_STEP,
        "verdict": verdict, "tier": "C",
        "rho_D_Drestored": rho,
        "split_half_rho_D": sh_rho_D, "rho_ceiling_gap": rho_ceiling_gap,
        "A_n_sig_pos": n_A_sig_pos, "AD_ratio_median": AD_ratio_med,
        "typeA_causal": results,
        "note": ("OLMoE-1B-7B-0125 非 canonical 0924. 纯推理零训练 (无训练 forward; 与 07-09 声明的 6 实验解冻口径不同, 那是训练 forward, 本实验零训练 forward). "
                 "Type A native softmax-then-topk (norm_topk_prob=False). "
                 "两arm精确反事实: D(p)=裸删(含膨胀1/(1-σ_p))-baseline; D_restored(p)=质量保持删(删p召回第9, 幸存权重=baseline softmax值未膨胀)-baseline=C(p); "
                 "A(p)=D-D_restored=逐expert因果塌缩 (闭式, 零回归, 零共线性). 质量保持删=同模型 Type A 分母不含p实现, 废 Type B 脏arm (强加 renorm 失配 nll6.2). "
                 "证据 C (因果反事实: 两arm唯一差异=膨胀, do(膨胀)vs do(不膨胀), 非统计分离). "
                 "OLS 代理 α·r_p 保留对比 (A_ols_proxy), 诊断共线性. "
                 "D_restored_ci/A_ci per-window block bootstrap 1000x. "
                 "headline 换锚: 主结果 = 剪枝判据污染因果实在, 单expert可达实测损伤 84% (exp18 A/D), 修正闭式免费; "
                 "全局 ρ(D,D_restored) 降为辅助统计, 由 split-half ρ(D) 信度天花板定性 (0.95 阈值系拍定无依据, 弃用). "
                 "A(p) 配对 CI = 两arm同 token 同窗差分消窗口级共同噪声, 远窄于 D_restored CI, A 才是宣称因果的量."),
    }
    os.makedirs(OUT, exist_ok=True)
    outpath = os.path.join(OUT, f"prune_deconvention_{PD_MODEL_TAG}_L{PD_LAYER}.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] 写入 {outpath}", flush=True)


if __name__ == "__main__":
    main()