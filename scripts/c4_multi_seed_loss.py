"""c4_multi_seed_loss.py — 多 seed confirm PAMS loss 是否真不增 (目标支点夯实)

frozen declaration: 训练 forward, 第 10 个, 超出 6 实验解冻范围, 用户授权解冻跑。
purpose: λ 扫描发现 λ=0.01 loss=2.108 < hard 2.366 (单 seed 末steps), 本轮多 seed + 末段平均
  confirm PAMS loss 不增是否跨 seed 稳健。这决定 PAMS 与 2.2 区分最后剩下的目标支点:
  - 若 pams 末段 loss ≤ hard 占多数 + mean delta < 0 => 目标支点夯实 (降翻转 + loss 不增/降)
  - 若方向对半 => loss 持平 (目标支点勉强)
  - 若 pams > hard 占多数 => loss 实际微增 (目标支点变弱, PAMS 退化为 2.2 风险)

design: 4 seed × (hard + pams λ=0.01) 配对, 同 seed 仅 routing 不同, 末段平均 loss
  (最后 5 个记录点 step 1500-1900, 消除末steps噪声) + A 翻转率。

output: results/c4_multi_seed_loss.json
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
OUT_PATH = OUT_DIR + "/c4_multi_seed_loss.json"
SEEDS = [0, 1, 2, 3]
PAMS_EPS = 0.008
PAMS_LAMBDA = 0.01


def end_segment_loss(losses, n_tail=5):
    # 末段平均: 最后 n_tail 个记录点的 loss 均值 (消除末steps噪声)
    tail = [lv for _, lv in losses[-n_tail:]]
    return sum(tail) / len(tail) if tail else None


def main():
    steps = int(os.environ.get("C4_STEPS", "2000"))
    base_args = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "d": 384, "n_head": 6, "n_layer": 6, "hidden": 1024,
        "n_exp": 8, "k": 2, "block": 256, "bs": 32,
        "steps": steps, "lr": 3e-3, "wd": 0.1, "grad_clip": 1.0, "amp": True,
    }
    tau0 = 1.0 / base_args["n_exp"]
    print(f"[mseed] seeds={SEEDS} steps={steps} pams λ={PAMS_LAMBDA} ε={PAMS_EPS}")
    res = {"config": base_args, "pams_eps": PAMS_EPS, "pams_lambda": PAMS_LAMBDA,
           "seeds": SEEDS, "per_seed": []}

    for seed in SEEDS:
        args = dict(base_args, seed=seed)
        print(f"[mseed] === seed {seed} ===")
        # hard arm
        h_model, val_data, _, h_losses, _ = c4.train_arm("hard", PAMS_EPS, 0.0, args)
        h_end = end_segment_loss(h_losses)
        h_a = c4.probe_arm(h_model, val_data, args, tau0)["A_full_forward_flip"]["flip_rate"]
        del h_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # pams λ=0.01 arm
        p_model, _, _, p_losses, _ = c4.train_arm("pams", PAMS_EPS, PAMS_LAMBDA, args)
        p_end = end_segment_loss(p_losses)
        p_a = c4.probe_arm(p_model, val_data, args, tau0)["A_full_forward_flip"]["flip_rate"]
        del p_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        delta = p_end - h_end if (p_end is not None and h_end is not None) else None
        flip_red = (h_a - p_a) / h_a if h_a > 0 else 0.0
        res["per_seed"].append({
            "seed": seed, "hard_end_loss": h_end, "pams_end_loss": p_end,
            "loss_delta": delta, "hard_A_flip": h_a, "pams_A_flip": p_a,
            "flip_reduction": flip_red,
        })
        print(f"[mseed] seed {seed}: hard_end={h_end:.4f} pams_end={p_end:.4f} "
              f"delta={delta:+.4f} | hard_A={h_a:.4f} pams_A={p_a:.4f} flip_red={flip_red:.3f}")

    # 配对统计
    deltas = [s["loss_delta"] for s in res["per_seed"] if s["loss_delta"] is not None]
    flip_reds = [s["flip_reduction"] for s in res["per_seed"]]
    n_pams_lower = sum(1 for d in deltas if d < 0)
    n_pams_higher = sum(1 for d in deltas if d > 0)
    mean_delta = sum(deltas) / len(deltas) if deltas else None
    mean_flip_red = sum(flip_reds) / len(flip_reds) if flip_reds else None
    res["summary"] = {
        "n_seeds": len(deltas), "n_pams_loss_lower": n_pams_lower,
        "n_pams_loss_higher": n_pams_higher, "mean_loss_delta": mean_delta,
        "mean_flip_reduction": mean_flip_red,
    }
    # verdict: 目标支点是否夯实
    if n_pams_lower >= 3 and mean_delta < 0:
        verdict = "LOSS_NOT_INCREASED_TARGET_HELD"  # 多数 seed pams loss < hard
    elif n_pams_lower + n_pams_higher == 0:
        verdict = "INCONCLUSIVE"
    elif n_pams_lower >= n_pams_higher and abs(mean_delta) < 0.05:
        verdict = "LOSS_HELD_MARGINAL"  # 持平勉强
    else:
        verdict = "LOSS_INCREASED_TARGET_WEAKENED"  # 多数 seed pams loss > hard, 目标支点变弱
    res["verdict"] = verdict
    res["evidence_tier"] = "E"
    res["note"] = ("多 seed 末段平均confirm PAMS loss 不增. "
                   "LOSS_NOT_INCREASED => 目标支点夯实 (与2.2区分剩目标支点成立); "
                   "LOSS_INCREASED => 目标支点变弱, PAMS 退化为2.2风险.")
    with open(OUT_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[mseed] === summary ===")
    print(f"  pams loss < hard: {n_pams_lower}/{len(deltas)} | > hard: {n_pams_higher}/{len(deltas)}")
    print(f"  mean loss_delta={mean_delta:+.4f} | mean flip_reduction={mean_flip_red:.3f}")
    print(f"[mseed] verdict={verdict}")
    print(f"[mseed] 写入 {OUT_PATH}")


if __name__ == "__main__":
    main()