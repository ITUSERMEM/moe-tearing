#!/bin/bash
# run_three_arms.sh — 三臂串行重跑, 每臂 20000 步
LOG_DIR="$(dirname "$0")/../logs"
mkdir -p "$LOG_DIR"

export PAMS_STEPS=20000
export PAMS_BS=4
export PAMS_LR=3e-4
export PAMS_TAU=0.02
export CUDA_VISIBLE_DEVICES=0

echo "[run] === Arm 1/3: λ=0.0 (baseline) ==="
PAMS_LAMBDA=0.0 PAMS_TAG=l0_0 python3 "$(dirname "$0")/granite_pams_align.py" 2>&1 | tee "$LOG_DIR/arm_l0_0.log"
echo "[run] === Arm 2/3: λ=0.1 ==="
PAMS_LAMBDA=0.1 PAMS_TAG=l0_1 python3 "$(dirname "$0")/granite_pams_align.py" 2>&1 | tee "$LOG_DIR/arm_l0_1.log"
echo "[run] === Arm 3/3: λ=1.0 ==="
PAMS_LAMBDA=1.0 PAMS_TAG=l1_0 python3 "$(dirname "$0")/granite_pams_align.py" 2>&1 | tee "$LOG_DIR/arm_l1_0.log"
echo "[run] === All done ==="
