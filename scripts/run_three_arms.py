"""run_three_arms.py — 三arm串行重跑

用法: python3 scripts/run_three_arms.py
日志: logs/arm_l{LAMBDA}.log
"""
import os, sys, subprocess, time

SCRIPT = os.path.join(os.path.dirname(__file__), "granite_pams_align.py")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

arms = [
    {"PAMS_LAMBDA": "0.0", "PAMS_TAG": "l0_0", "label": "λ=0.0 (baseline)"},
    {"PAMS_LAMBDA": "0.1", "PAMS_TAG": "l0_1", "label": "λ=0.1"},
    {"PAMS_LAMBDA": "1.0", "PAMS_TAG": "l1_0", "label": "λ=1.0"},
]

env = os.environ.copy()
env.update({"PAMS_STEPS": "20000", "PAMS_BS": "4", "PAMS_LR": "3e-4", "PAMS_TAU": "0.02", "CUDA_VISIBLE_DEVICES": "0"})

for i, arm in enumerate(arms):
    label = arm["label"]
    log_path = os.path.join(LOG_DIR, f"arm_{arm['PAMS_TAG']}.log")
    print(f"\n[run] === Arm {i+1}/3: {label} ===", flush=True)
    env.update({"PAMS_LAMBDA": arm["PAMS_LAMBDA"], "PAMS_TAG": arm["PAMS_TAG"]})
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run([sys.executable, SCRIPT], env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = (time.time() - t0) / 3600
    print(f"[run] {label} done: {'OK' if proc.returncode==0 else f'FAIL (code={proc.returncode})'}, {elapsed:.1f}h", flush=True)
    if proc.returncode != 0:
        print(f"[run] log tail: {open(log_path).read().strip()[-500:]}", flush=True)

print(f"\n[run] === All three arms complete ===", flush=True)
