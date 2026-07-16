"""run_low_lambda.py — 低 λ BEAT LoRA 微调四arm串行"""

import os, sys, subprocess, time

SCRIPT = os.path.join(os.path.dirname(__file__), "olmoe_beat_lora.py")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
STEPS = 2000

arms = [
    (0.0,    "lora_l0_baseline"),
    (0.05,   "lora_l0_05"),
    (0.01,   "lora_l0_01"),
    (0.005,  "lora_l0_005"),
]

env = os.environ.copy()
env.update({"OLLM_STEPS": str(STEPS), "CUDA_VISIBLE_DEVICES": "0"})

for lam, tag in arms:
    log_path = os.path.join(LOG_DIR, f"low_lambda_{tag}.log")
    env.update({"OLLM_LAMBDA": str(lam), "OLLM_TAG": tag})
    t0 = time.time()
    print(f"\n[run] λ={lam} steps={STEPS} tag={tag}", flush=True)
    with open(log_path, "w") as log:
        ret = subprocess.run([sys.executable, SCRIPT], env=env, stdout=log, stderr=subprocess.STDOUT)
    t = (time.time() - t0) / 60
    status = "OK" if ret.returncode == 0 else f"FAIL(code={ret.returncode})"
    print(f"[run] λ={lam} {status} {t:.1f}min", flush=True)
    if ret.returncode != 0:
        with open(log_path) as f:
            print(f"  tail: {f.read().strip()[-300:]}", flush=True)

print("\n[run] All done", flush=True)
