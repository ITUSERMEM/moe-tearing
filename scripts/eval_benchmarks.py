"""eval_benchmarks.py — 合并 LoRA + lm-eval 基准测试

用法: python3 scripts/eval_benchmarks.py
"""

import os, sys, json, subprocess, time
import torch
from transformers import AutoModelForCausalLM
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "..", "weights")
MODEL_ID = "allenai/OLMoE-1B-7B-0125"
DEVICE = "cuda"

def merge_and_save(tag, lora_path, save_path):
    """Merge LoRA weights into base model and save"""
    if os.path.exists(os.path.join(save_path, "model.safetensors")):
        print(f"[merge] {tag}: already exists at {save_path}, skip", flush=True)
        return
    print(f"[merge] {tag}: loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE, torch_dtype=torch.bfloat16)
    sd = torch.load(lora_path, map_location=DEVICE)
    moe_blocks = [(n,m) for n,m in model.named_modules() if isinstance(m, OlmoeSparseMoeBlock)]
    for li, (_, blk) in enumerate(moe_blocks):
        ag = sd[f"L{li}_A_gate"].float()
        bg = sd[f"L{li}_B_gate"].float()
        ad = sd[f"L{li}_A_down"].float()
        bd = sd[f"L{li}_B_down"].float()
        gu = blk.experts.gate_up_proj.data.float()
        dn = blk.experts.down_proj.data.float()
        for e in range(gu.shape[0]):
            gu[e] = gu[e] + (ag[e] @ bg[e]).to(gu.dtype)
            dn[e] = dn[e] + (ad[e] @ bd[e]).to(dn.dtype)
        blk.experts.gate_up_proj.data = gu.to(torch.bfloat16)
        blk.experts.down_proj.data = dn.to(torch.bfloat16)
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path, safe_serialization=True)
    print(f"[merge] {tag}: saved to {save_path}", flush=True)
    del model
    torch.cuda.empty_cache()

def run_lmeval(model_path, tag, tasks="mmlu,hellaswag,gsm8k", fewshot=5):
    """Run lm-eval on a merged model"""
    out_path = os.path.join(OUT_DIR, f"benchmark_{tag}.json")
    if os.path.exists(out_path):
        print(f"[eval] {tag}: results already exist at {out_path}, skip", flush=True)
        with open(out_path) as f:
            return json.load(f)
    print(f"[eval] {tag}: running tasks={tasks}...", flush=True)
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=bfloat16",
        "--tasks", tasks,
        "--num_fewshot", str(fewshot),
        "--batch_size", "4",
        "--output_path", out_path,
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = (time.time() - t0) / 60
    print(f"[eval] {tag}: done in {elapsed:.1f}min, returncode={result.returncode}", flush=True)
    if result.returncode != 0:
        print(f"[eval] {tag} stderr: {result.stderr[-500:]}", flush=True)
    # Try to load results from lm-eval output
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    return None

def main():
    arms = {
        "baseline": {
            "lora": os.path.join(OUT_DIR, "olmoe_beat_lora_lora_l0_baseline_lora.pt"),
            "merged": os.path.join(WEIGHTS_DIR, "olmoe_beat_lora_baseline_merged"),
        },
        "beat_0.05": {
            "lora": os.path.join(OUT_DIR, "olmoe_beat_lora_lora_l0_05_lora.pt"),
            "merged": os.path.join(WEIGHTS_DIR, "olmoe_beat_lora_0.05_merged"),
        },
    }

    # Step 1: Merge LoRA weights
    for tag, cfg in arms.items():
        merge_and_save(tag, cfg["lora"], cfg["merged"])

    # Step 2: Run benchmarks (MMLU 5-shot, HellaSwag 10-shot, GSM8K 8-shot)
    results = {}
    for tag, cfg in arms.items():
        r = run_lmeval(cfg["merged"], tag, tasks="mmlu,hellaswag,gsm8k", fewshot=5)
        if r:
            results[tag] = r
            print(f"\n[result] {tag}:", flush=True)
            for task in ["mmlu", "hellaswag", "gsm8k"]:
                if "results" in r and task in r["results"]:
                    acc = r["results"][task].get("acc,none", r["results"][task].get("acc","?"))
                    print(f"  {task}: {acc}", flush=True)

    # Save summary
    summary = {}
    for tag, r in results.items():
        summary[tag] = {}
        if r and "results" in r:
            for task in ["mmlu", "hellaswag", "gsm8k"]:
                if task in r["results"]:
                    summary[tag][task] = r["results"][task]
    summary_path = os.path.join(OUT_DIR, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] summary -> {summary_path}", flush=True)

if __name__ == "__main__":
    main()
