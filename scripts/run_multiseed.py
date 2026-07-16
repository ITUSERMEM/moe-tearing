"""run_multiseed.py — 多 seed 夜跑 (λ=0/1.0 × seed 43,44,45 × 10k steps)
"""

import os, sys, subprocess, time, json
import torch, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(SCRIPT_DIR, "granite_pams_align.py")
LOG_DIR = os.path.join(SCRIPT_DIR, "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

STEPS = 10000
SEEDS = [43, 44, 45]
ARMS = [(0.0, "l0"), (1.0, "l10")]

env = os.environ.copy()
env.update({"PAMS_STEPS": str(STEPS), "PAMS_BS": "4", "PAMS_LR": "3e-4",
            "PAMS_TAU": "0.02", "CUDA_VISIBLE_DEVICES": "0"})

for lam, tag_base in ARMS:
    for seed in SEEDS:
        tag = f"{tag_base}_s{seed}"
        log_path = os.path.join(LOG_DIR, f"multiseed_{tag}.log")
        env.update({"PAMS_LAMBDA": str(lam), "PAMS_TAG": tag, "PAMS_SEED": str(seed)})
        t0 = time.time()
        print(f"\n[run] λ={lam} seed={seed} start", flush=True)
        with open(log_path, "w") as log:
            ret = subprocess.run([sys.executable, MODEL], env=env, stdout=log, stderr=subprocess.STDOUT)
        t = (time.time() - t0) / 3600
        status = "OK" if ret.returncode == 0 else f"FAIL(code={ret.returncode})"
        print(f"[run] λ={lam} seed={seed} {status} {t:.1f}h", flush=True)
        if ret.returncode != 0:
            with open(log_path) as f:
                print(f"  last lines: {f.read().strip()[-300:]}", flush=True)

print(f"\n[run] All 6 arms submitted!", flush=True)  # Jaccard zero-line runs after all arms complete
