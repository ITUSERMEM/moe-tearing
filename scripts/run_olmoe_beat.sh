#!/bin/bash
# run_olmoe_beat.sh — OLMoE BEAT FT L8 三臂串行
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"

export OLLM_STEPS=200
export OLLM_BS=2
export OLLM_LR=1e-4
export CUDA_VISIBLE_DEVICES=0

echo "[run] === OLMoE BEAT FT L8 ==="
echo "[run] Arm 1/3: λ=0.0 (baseline)"
OLLM_LAMBDA=0.0 OLLM_TAG=l0_0 python3 "$SCRIPT_DIR/olmoe_beat_ft.py" 2>&1 | tee "$LOG_DIR/olmoe_l0_0.log"

echo "[run] Arm 2/3: λ=0.1"
OLLM_LAMBDA=0.1 OLLM_TAG=l0_1 python3 "$SCRIPT_DIR/olmoe_beat_ft.py" 2>&1 | tee "$LOG_DIR/olmoe_l0_1.log"

echo "[run] Arm 3/3: λ=1.0"
OLLM_LAMBDA=1.0 OLLM_TAG=l1_0 python3 "$SCRIPT_DIR/olmoe_beat_ft.py" 2>&1 | tee "$LOG_DIR/olmoe_l1_0.log"

echo "[run] === All OLMoE arms complete ==="
