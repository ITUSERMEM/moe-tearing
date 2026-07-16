"""c4_pams_vs_hard.py — C4 训练对比: PAMS vs hard (核心命题verification)

frozen declaration: 训练 forward (新), 第 8 个 forward, 超出 08 methodology 6 实验解冻范围,
  用户授权解冻跑。两arm同 seed/data/steps, 仅 routing 不同。

核心命题 (PAMS 不被 Sec8 反杀的实证支点):
  PAMS 降 bf16 routing翻转率 (A 全链路口径) 而 loss 持平。若翻转降 + loss 持平 =>
  PAMS_REDUCES_FLIP_LOSS_HELD (PAMS 通过 Sec8 一致性检查); 若翻转降但 loss 显著升 =>
  PAMS_REDUCES_FLIP_LOSS_COST (退化为 2.2 风险); 若翻转不降 => PAMS_NO_EFFECT (无对象).

measurement (每arm训练后, 纯推理):
  - A 全模型 bf16 链路翻转率 (复用 preflight probe_a)
  - C d_t = z_[k]-z_[k+1] 左尾分布 (复用 preflight probe_c, PAMS 作用集)
  - final loss (CE, pams arm含regularization贡献但报 CE loss)
  - M2/hardG/hardJump (mtr.probe_tear, C2 预测: 拓扑应unchanged, 相对差异看 pams vs hard)

output: results/c4_pams_vs_hard.json
环境: MTEAR_PROBE_DIR=/path/to/project/moe-tearing-main (必需)
  C4_STEPS (默认 2000), C4_PAMS_EPS (默认 0.008), C4_PAMS_LAMBDA (默认 0.1) 可调
"""
import sys
import os
import json
import math

PROBE_DIR = os.environ["MTEAR_PROBE_DIR"]
sys.path.insert(0, PROBE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 复用 preflight 探针
import torch
import moe_train_probe as mtr
import preflight_probe_bf16_flip as pre

OUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../results"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = OUT_DIR + "/c4_pams_vs_hard.json"


def train_arm(routing, pams_eps, pams_lambda, args):
    # 两arm同 seed/data/steps, 仅 routing 不同; pams arm加 pams_lambda * pams_reg 到 loss
    device = args["device"]
    gen = torch.Generator().manual_seed(args["seed"])
    torch.manual_seed(args["seed"])
    train_data, val_data, vocab = mtr.load_char(None)[:3]
    tau0 = 1.0 / args["n_exp"]
    model = mtr.GPTMoE(vocab, args["d"], args["n_head"], args["n_layer"], args["hidden"],
                      args["n_exp"], args["k"], routing, tau0, args["block"]).to(device)
    if routing == "pams":
        model.set_pams_eps(pams_eps)
    opt = torch.optim.AdamW(model.parameters(), lr=args["lr"], betas=(0.9, 0.95),
                            weight_decay=args["wd"])
    use_amp = args["amp"] and device == "cuda"
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    losses = []
    pams_reg_curve = []
    for step in range(args["steps"]):
        x, y = mtr.get_batch(train_data, args["block"], args["bs"], device, gen)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss, l1, zloss, _ = model(x, y)
            total = loss + (pams_lambda * l1 if routing == "pams" else 0.0)
        if not math.isfinite(loss.item()):
            print(f"[c4] {routing} step {step}: loss non-finite -> STOP")
            break
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
        opt.step()
        if step % max(1, args["steps"] // 20) == 0:
            losses.append((step, round(loss.item(), 4)))
            if routing == "pams":
                pams_reg_curve.append((step, round(l1.item(), 4)))
            print(f"[c4] {routing} step {step} loss={loss.item():.4f}"
                  + (f" pams_reg={l1.item():.4f}" if routing == "pams" else ""))
    return model, val_data, vocab, losses, pams_reg_curve


def probe_arm(model, val_data, args, tau0):
    # 训练后纯推理: A 翻转率 + C d_t 左尾 + M2/hardG unchanged性
    k = args["k"]
    gen = torch.Generator().manual_seed(2026)
    idx, _ = mtr.get_batch(val_data, args["block"], 256, args["device"], gen)
    res = {}
    res.update(pre.probe_a_full_forward_flip(model, idx, k))
    res.update(pre.probe_c_dt_distribution(model, idx, k))
    # M2/hardG (probe_tear, 小 batch 防 OOM)
    probe_x = idx[:16]
    try:
        tear = mtr.probe_tear(model, probe_x, k, tau0, layer=-1)
        res["M2_tear_med"] = tear["tear_med"]
        res["M2_cos_kk1"] = tear["cos_kk1"]
        res["hardG"] = tear["hardG"]
        res["hardJump"] = tear["hardJump"]
    except Exception as e:
        res["tear_error"] = str(e)
    return res


def main():
    steps = int(os.environ.get("C4_STEPS", "2000"))
    pams_eps = float(os.environ.get("C4_PAMS_EPS", "0.008"))
    pams_lambda = float(os.environ.get("C4_PAMS_LAMBDA", "0.1"))
    args = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 0, "d": 384, "n_head": 6, "n_layer": 6, "hidden": 1024,
        "n_exp": 8, "k": 2, "block": 256, "bs": 32,
        "steps": steps, "lr": 3e-3, "wd": 0.1, "grad_clip": 1.0, "amp": True,
    }
    tau0 = 1.0 / args["n_exp"]
    print(f"[c4] config steps={steps} pams_eps={pams_eps} pams_lambda={pams_lambda}")
    res = {"config": args, "pams_eps": pams_eps, "pams_lambda": pams_lambda, "arms": {}}
    for routing in ("hard", "pams"):
        print(f"[c4] === 训练 {routing} arm ===")
        model, val_data, vocab, losses, pams_reg_curve = train_arm(
            routing, pams_eps, pams_lambda, args)
        arm = {"final_loss": losses[-1][1] if losses else None,
               "train_loss_curve": losses}
        if routing == "pams":
            arm["pams_reg_curve"] = pams_reg_curve
        arm.update(probe_arm(model, val_data, args, tau0))
        res["arms"][routing] = arm
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # verdict核心命题
    h = res["arms"]["hard"]
    p = res["arms"]["pams"]
    h_a = h["A_full_forward_flip"]["flip_rate"]
    p_a = p["A_full_forward_flip"]["flip_rate"]
    h_loss = h["final_loss"]
    p_loss = p["final_loss"]
    flip_reduction = (h_a - p_a) / h_a if h_a > 0 else 0.0
    loss_delta = (p_loss - h_loss) if (h_loss is not None and p_loss is not None) else None
    res["flip_reduction_frac"] = flip_reduction
    res["loss_delta"] = loss_delta
    # verdict: 翻转降 >20% 且 loss delta < 10% of hard loss => 通过 Sec8 一致性
    loss_held = abs(loss_delta) < 0.10 * abs(h_loss) if loss_delta is not None else False
    if flip_reduction > 0.2 and loss_held:
        verdict = "PAMS_REDUCES_FLIP_LOSS_HELD"
    elif flip_reduction > 0.2 and not loss_held:
        verdict = "PAMS_REDUCES_FLIP_LOSS_COST"
    elif flip_reduction > 0.05:
        verdict = "PAMS_WEAK_REDUCTION"
    else:
        verdict = "PAMS_NO_EFFECT"
    res["verdict"] = verdict
    res["evidence_tier"] = "E"
    res["note"] = ("C4 核心命题: PAMS 降 bf16 翻转率而 loss 持平 (Sec8 一致性). "
                   "REDUCES_FLIP_LOSS_HELD=通过; LOSS_COST=退化为2.2风险; "
                   "NO_EFFECT=PAMS 无对象 (反数据修补: 如实报告不重写).")
    with open(OUT_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[c4] verdict={verdict}")
    print(f"[c4] hard A_flip={h_a:.4f} loss={h_loss} | pams A_flip={p_a:.4f} loss={p_loss}")
    print(f"[c4] flip_reduction={flip_reduction:.3f} loss_delta={loss_delta}")
    if "hardG" in h and "hardG" in p:
        print(f"[c4] hardG: hard={h['hardG']} pams={p['hardG']} (C2 拓扑unchanged预测)")
    print(f"[c4] 写入 {OUT_PATH}")


if __name__ == "__main__":
    main()