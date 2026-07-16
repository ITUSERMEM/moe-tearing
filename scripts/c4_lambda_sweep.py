"""c4_lambda_sweep.py — PAMS λ 扫描: confirm"全局 margin push up"是机制还是 λ 过强伪影

frozen declaration: 训练 forward, 第 9 个 forward, 超出 6 实验解冻范围, 用户授权解冻跑。
purpose (回答 C3 是否可救): C4 首轮 λ=0.1 发现 PAMS 后 d_t 中位 0.886->10.58 (12×),
  C3"稀疏下尾裁剪"论点证伪。本扫 λ 看弱 λ 是否只裁下尾不推全局:
  - 若弱 λ 下 d_t 中位接近 hard (不全局push up) + A 翻转仍降 + loss 持平 => C3 救回
  - 若弱 λ 仍 d_t 中位 >> hard => C3 彻底证伪, PAMS 实质全局 margin 整形

design: hard baseline跑一次, pams 扫 λ ∈ {0.01, 0.03, 0.1, 0.3}, ε=0.008 固定,
  每arm 2000 steps同 seed/data。测 A 翻转率 + d_t 中位/1%分位/frac<0.01 + loss + hardG。

output: results/c4_lambda_sweep.json
环境: MTEAR_PROBE_DIR=/path/to/project/moe-tearing-main (必需), C4_STEPS (默认 2000)
"""
import sys
import os
import json
import math

PROBE_DIR = os.environ["MTEAR_PROBE_DIR"]
sys.path.insert(0, PROBE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import moe_train_probe as mtr
import c4_pams_vs_hard as c4

OUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../results"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = OUT_DIR + "/c4_lambda_sweep.json"
LAMBDAS = [0.01, 0.03, 0.1, 0.3]


def main():
    steps = int(os.environ.get("C4_STEPS", "2000"))
    pams_eps = 0.008
    args = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 0, "d": 384, "n_head": 6, "n_layer": 6, "hidden": 1024,
        "n_exp": 8, "k": 2, "block": 256, "bs": 32,
        "steps": steps, "lr": 3e-3, "wd": 0.1, "grad_clip": 1.0, "amp": True,
    }
    tau0 = 1.0 / args["n_exp"]
    print(f"[sweep] steps={steps} eps={pams_eps} lambdas={LAMBDAS}")
    res = {"config": args, "pams_eps": pams_eps, "lambdas": LAMBDAS, "arms": {}}

    # hard baseline (跑一次)
    print("[sweep] === hard baseline ===")
    h_model, val_data, vocab, h_losses, _ = c4.train_arm("hard", pams_eps, 0.0, args)
    h_probe = c4.probe_arm(h_model, val_data, args, tau0)
    res["arms"]["hard"] = {"final_loss": h_losses[-1][1] if h_losses else None,
                           **h_probe}
    del h_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # pams 扫 λ
    for lam in LAMBDAS:
        print(f"[sweep] === pams λ={lam} ===")
        p_model, _, _, p_losses, reg_curve = c4.train_arm("pams", pams_eps, lam, args)
        p_probe = c4.probe_arm(p_model, val_data, args, tau0)
        res["arms"][f"pams_lam{lam}"] = {
            "lambda": lam, "final_loss": p_losses[-1][1] if p_losses else None,
            "pams_reg_final": reg_curve[-1][1] if reg_curve else None,
            **p_probe,
        }
        del p_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # verdict C3 是否可救
    h_dt_med = res["arms"]["hard"]["C_dt_distribution"]["d_t_50pct"]
    h_a = res["arms"]["hard"]["A_full_forward_flip"]["flip_rate"]
    h_loss = res["arms"]["hard"]["final_loss"]
    sweep_table = []
    for lam in LAMBDAS:
        a = res["arms"][f"pams_lam{lam}"]
        dt_med = a["C_dt_distribution"]["d_t_50pct"]
        dt_1p = a["C_dt_distribution"]["d_t_1pct"]
        frac001 = a["C_dt_distribution"]["left_tail_frac"]["frac_d_t_lt_0.01"]
        af = a["A_full_forward_flip"]["flip_rate"]
        loss = a["final_loss"]
        hardg = a.get("hardG")
        dt_inflate = dt_med / h_dt_med if h_dt_med > 0 else float("inf")
        flip_red = (h_a - af) / h_a if h_a > 0 else 0.0
        loss_delta = loss - h_loss if (loss is not None and h_loss is not None) else None
        sweep_table.append({"lambda": lam, "dt_median": dt_med, "dt_inflate_x": dt_inflate,
                            "dt_1pct": dt_1p, "frac_dt_lt_0.01": frac001,
                            "A_flip": af, "flip_reduction": flip_red,
                            "loss": loss, "loss_delta": loss_delta, "hardG": hardg})
    res["sweep_table"] = sweep_table

    # verdict: 弱 λ (0.01) 下 dt_inflate 是否接近 1
    weak = res["arms"]["pams_lam0.01"]["C_dt_distribution"]["d_t_50pct"] / h_dt_med
    weak_flip = (h_a - res["arms"]["pams_lam0.01"]["A_full_forward_flip"]["flip_rate"]) / h_a if h_a > 0 else 0
    if weak < 1.5 and weak_flip > 0.2:
        verdict = "C3_RESCUABLE_WEAK_LAMBDA"  # 弱 λ 只裁下尾不推全局
    elif weak < 1.5 and weak_flip <= 0.2:
        verdict = "C3_RESCUABLE_BUT_NO_FLIP_REDUCTION"  # 弱 λ 不推全局但也不降翻转
    else:
        verdict = "C3_FALSIFIED_GLOBAL_INFLATION"  # 弱 λ 仍全局push up
    res["verdict"] = verdict
    res["evidence_tier"] = "E"
    res["note"] = ("C3 是否可救: 看弱 λ=0.01 下 d_t 中位膨胀倍数. "
                   "RESCUABLE => 弱 λ 只裁下尾 (C3 论点救回); "
                   "FALSIFIED_GLOBAL_INFLATION => PAMS 实质全局 margin 整形 (C3 彻底证伪).")
    with open(OUT_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print("[sweep] === verdict表 ===")
    print(f"{'λ':<6}{'dt_med':<10}{'dt_×':<8}{'A_flip':<10}{'flip_red':<10}{'loss':<10}{'hardG':<8}")
    for r in sweep_table:
        print(f"{r['lambda']:<6}{r['dt_median']:<10.3f}{r['dt_inflate_x']:<8.2f}"
              f"{r['A_flip']:<10.4f}{r['flip_reduction']:<10.3f}{str(r['loss']):<10}{str(r['hardG']):<8}")
    print(f"[sweep] hard baseline dt_med={h_dt_med:.3f} A_flip={h_a:.4f} loss={h_loss}")
    print(f"[sweep] verdict={verdict} (弱λ=0.01 dt膨胀={weak:.2f}×, 翻转降={weak_flip:.3f})")
    print(f"[sweep] 写入 {OUT_PATH}")


if __name__ == "__main__":
    main()