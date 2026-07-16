"""preflight_probe_bf16_flip.py — PAMS 前置verification: from-scratch probe bf16 量化翻转现象复现性

frozen declaration (重要):
  含训练 forward (新), 超出 08 methodology 已声明的 6 实验解冻范围 (①-⑥)。
  须单独解冻授权后跑。训练复用论文 Sec7/8 已 established 的 from-scratch probe 范式
  (GPTMoE + load_char + bf16 autocast), 训练后只跑纯推理探针, 不改训练逻辑。

purpose (PAMS 可行性前置, 对应技术路线图 v2 §5 "前置verification"):
  C'/exp2 在真模型 (OLMoE/Qwen) 上发现 bf16 在routingboundary附近产生假翻转; exp2 verdict
  FP32_NOISE_FLOOR_GAP_INFLATED_BF16 量化测了 bf16 噪声底。PAMS 危害模型
  "低精度噪声 x 高boundary密度 = routing非确定性" 依赖此现象。前置verification问: from-scratch probe
  训练后是否复现该现象? 若不复现, PAMS 在 probe 载体上无对象, 需重评估载体选择。

三个measurement (训练后, 纯推理, 无梯度):
  A. 全模型 bf16 链路翻转: 同input idx, fp32 全 forward vs bf16 权重全 forward,
     last-block top-k 集翻转率 (C'/exp2 真模型口径, 权重 bf16)
  B. 单层 gate 量化翻转: 固定 fp32 抓的 h, fp32 gate(h) vs bf16 gate(h) top-k 翻转率
     (PAMS 作用点隔离, 排除前面层累积舍入; B1 autocast 权重fp32 / B2 权重bf16 两口径)
  C. d_t = z_[k]-z_[k+1] 分布左尾: M1 "住在boundary上" verification + epsilon 标定input
     (d_t 是 PAMS 阈值作用的 logit 空间量, 测其分布是 epsilon 标定前提)

output: results/preflight_probe_bf16_flip.json
环境: MTEAR_PROBE_DIR=/path/to/project/moe-tearing-main (必需, import moe_train_probe)
  GPU 推荐 (cuda), CPU 可跑但慢。numpy<2 pin。
"""
import sys
import os
import json
import math

PROBE_DIR = os.environ["MTEAR_PROBE_DIR"]  # 必需, 缺失快速失败
sys.path.insert(0, PROBE_DIR)

import torch
import torch.nn.functional as F
import moe_train_probe as mtr

OUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../results"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = OUT_DIR + "/preflight_probe_bf16_flip.json"


def topk_set(logits, k):
    # hard routing top-k 集由 logits.topk 决定 (softmax 单调, line 152-153)
    return logits.topk(k, dim=-1).indices  # (N, k)


def flip_stats(set_a, set_b):
    # 每 token: top-k 集任一位不同算翻转; 也报平均集合距离
    N, k = set_a.shape
    flipped = (set_a != set_b).any(dim=-1)  # (N,)
    flip_rate = flipped.float().mean().item()
    # Jaccard 距离: |对称差| / |并|, 每 token
    jacc = []
    for i in range(N):
        a = set(set_a[i].tolist())
        b = set(set_b[i].tolist())
        union = a | b
        jac = 1.0 - len(a & b) / len(union) if union else 0.0
        jacc.append(jac)
    return {"flip_rate": flip_rate, "jaccard_mean": sum(jacc) / len(jacc),
            "n_tokens": N, "n_flipped": int(flipped.sum().item())}


def grab_last_moe_input(model, idx):
    # hook 抓最后 block 的 moe input h (全 forward 后), 返回 h (fp32 detach)
    grab = {}
    blk = model.blocks[-1].moe
    h = blk.register_forward_hook(lambda m, i, o: grab.__setitem__("x", i[0].detach()))
    with torch.no_grad():
        model(idx)
    h.remove()
    x = grab["x"]
    return x.reshape(-1, x.shape[-1])  # (B,T,C) -> (N, C) 与 probe_tear 一致


def probe_a_full_forward_flip(model, idx, k):
    # A. 全模型 bf16 链路翻转: fp32 全 forward vs bf16 权重全 forward
    # 真 bf16 部署口径: 权重转 bf16 (对应 OLMoE/Qwen model.bfloat16() 装载)
    model.eval()
    with torch.no_grad():
        # fp32 全 forward
        model.float()
        h_fp = grab_last_moe_input(model, idx)
        blk = model.blocks[-1].moe
        z_fp = blk.gate(h_fp.float())
        set_fp = topk_set(z_fp, k)
        # bf16 权重全 forward
        model.bfloat16()
        h_bf = grab_last_moe_input(model, idx)
        z_bf = blk.gate(h_bf.bfloat16())
        set_bf = topk_set(z_bf, k)
        model.float()  # 恢复
    return {"A_full_forward_flip": flip_stats(set_fp, set_bf)}


def probe_b_gate_only_flip(model, idx, k):
    # B. 单层 gate 量化翻转: 固定 fp32 抓的 h, 隔离 gate 单层量化
    model.eval()
    with torch.no_grad():
        model.float()
        h_fp = grab_last_moe_input(model, idx).float()  # 固定参考 h
        blk = model.blocks[-1].moe
        z_fp = blk.gate(h_fp)  # fp32 gate
        set_fp = topk_set(z_fp, k)
        # B1 autocast: 权重 fp32, autocast bf16 matmul (训练时口径)
        with torch.autocast(device_type="cuda" if idx.is_cuda else "cpu",
                            dtype=torch.bfloat16):
            z_bf1 = blk.gate(h_fp)
        set_bf1 = topk_set(z_bf1.float(), k)
        # B2 权重 bf16: 量化部署口径
        gate_bf = blk.gate.bfloat16()
        z_bf2 = gate_bf(h_fp.bfloat16())
        set_bf2 = topk_set(z_bf2.float(), k)
    return {"B1_autocast_gate_flip": flip_stats(set_fp, set_bf1),
            "B2_weightbf16_gate_flip": flip_stats(set_fp, set_bf2)}


def probe_c_dt_distribution(model, idx, k):
    # C. d_t = z_[k]-z_[k+1] 分布左尾 (M1 boundary密度 + epsilon 标定input)
    model.eval()
    with torch.no_grad():
        model.float()
        h_fp = grab_last_moe_input(model, idx).float()
        z = model.blocks[-1].moe.gate(h_fp)  # (N, n_exp) fp32
        z_sorted = z.sort(dim=-1, descending=True).values
        d_t = z_sorted[:, k - 1] - z_sorted[:, k]  # 第k大 - 第k+1大 (N,)
        d_np = d_t.cpu().numpy()
    n = len(d_np)
    d_sorted = sorted(d_np)
    def pct(p):
        return float(d_sorted[max(0, int(p * n) - 1)])
    # 左尾质量: d_t < 候选 epsilon 的比例 (epsilon 标定input)
    eps_grid = [1e-4, 1e-3, 1e-2, 0.04, 0.1, 0.5]
    left_tail = {f"frac_d_t_lt_{e}": float((d_np < e).mean()) for e in eps_grid}
    return {"C_dt_distribution": {
        "n_tokens": n, "k": k,
        "d_t_min": float(d_sorted[0]), "d_t_1pct": pct(0.01),
        "d_t_5pct": pct(0.05), "d_t_50pct": pct(0.50), "d_t_mean": float(d_np.mean()),
        "left_tail_frac": left_tail,
    }}


def train_probe(args):
    # 复用论文 Sec7/8 from-scratch probe 范式: GPTMoE + load_char + bf16 autocast
    device = args["device"]
    gen = torch.Generator().manual_seed(args["seed"])
    torch.manual_seed(args["seed"])
    train_data, val_data, vocab = mtr.load_char(None)[:3]
    print(f"[preflight] data=char vocab={vocab} train_tokens={len(train_data)} device={device}")
    tau0 = 1.0 / args["n_exp"]
    model = mtr.GPTMoE(vocab, args["d"], args["n_head"], args["n_layer"], args["hidden"],
                      args["n_exp"], args["k"], "hard", tau0, args["block"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[preflight] model {n_params/1e6:.1f}M n_exp={args['n_exp']} k={args['k']}")
    opt = torch.optim.AdamW(model.parameters(), lr=args["lr"], betas=(0.9, 0.95),
                            weight_decay=args["wd"])
    use_amp = args["amp"] and device == "cuda"
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[preflight] bf16 autocast ON")
    losses = []
    for step in range(args["steps"]):
        x, y = mtr.get_batch(train_data, args["block"], args["bs"], device, gen)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss, _, _, _ = model(x, y)
        if not math.isfinite(loss.item()):
            print(f"[preflight] step {step}: loss non-finite -> STOP")
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
        opt.step()
        if step % max(1, args["steps"] // 20) == 0:
            losses.append((step, round(loss.item(), 4)))
            print(f"[preflight] step {step} loss={loss.item():.4f}")
    return model, val_data, vocab, losses


def main():
    args = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 0, "d": 384, "n_head": 6, "n_layer": 6, "hidden": 1024,
        "n_exp": 8, "k": 2, "block": 256, "bs": 32,
        "steps": int(os.environ.get("PREFLIGHT_STEPS", "2000")),
        "lr": 3e-3, "wd": 0.1, "grad_clip": 1.0, "amp": True,
    }
    print(f"[preflight] args={args}")
    print(f"[preflight] 训练 forward start (新 forward, 须单独解冻授权; 默认 smoke {args['steps']} steps)")
    model, val_data, vocab, losses = train_probe(args)
    gen = torch.Generator().manual_seed(2026)  # 探针用独立 seed
    idx, _ = mtr.get_batch(val_data, args["block"], 256, args["device"], gen)
    k = args["k"]
    print(f"[preflight] 训练done, 跑 bf16/fp32 量化翻转探针 (k={k})")
    res = {
        "config": args, "train_loss_curve": losses,
        "final_loss": losses[-1][1] if losses else None,
        "probe": {},
    }
    res["probe"].update(probe_a_full_forward_flip(model, idx, k))
    res["probe"].update(probe_b_gate_only_flip(model, idx, k))
    res["probe"].update(probe_c_dt_distribution(model, idx, k))
    # verdict: A/B 翻转率是否 > 0 (现象复现), C 左尾是否密集 (boundary密度)
    a_fr = res["probe"]["A_full_forward_flip"]["flip_rate"]
    b1_fr = res["probe"]["B1_autocast_gate_flip"]["flip_rate"]
    b2_fr = res["probe"]["B2_weightbf16_gate_flip"]["flip_rate"]
    frac_001 = res["probe"]["C_dt_distribution"]["left_tail_frac"]["frac_d_t_lt_0.04"]
    if a_fr > 0.01 or b2_fr > 0.01:
        verdict = "PHENOMENON_REPRODUCED"
    elif a_fr > 0 or b2_fr > 0:
        verdict = "PHENOMENON_WEAK"
    else:
        verdict = "PHENOMENON_ABSENT"
    res["verdict"] = verdict
    res["evidence_tier"] = "E"
    res["note"] = ("前置verification: from-scratch probe 是否复现 bf16 量化routing翻转. "
                   "PHENOMENON_ABSENT => PAMS 在 probe 载体无对象, 需重评估.")
    with open(OUT_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[preflight] verdict={verdict} A_flip={a_fr:.4f} B1={b1_fr:.4f} B2={b2_fr:.4f} "
          f"frac_d_t<0.04={frac_001:.4f}")
    print(f"[preflight] 写入 {OUT_PATH}")


if __name__ == "__main__":
    main()