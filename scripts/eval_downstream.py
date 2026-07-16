"""eval_downstream.py — OLMoE downstream eval (HellaSwag + PIQA)

用法:
  python3 scripts/eval_downstream.py baseline         # 原模型
  python3 scripts/eval_downstream.py lora_merged      # LoRA 合并
  python3 scripts/eval_downstream.py both             # 两个都跑
"""

import os, sys, json, torch, torch.nn.functional as F, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
OUT_PATH = os.path.join(OUT_DIR, "downstream_results.json")
MERGE_DIR = os.path.join(SCRIPT_DIR, "..", "weights", "olmoe_beat_lora_merged")
MODEL_ID = "allenai/OLMoE-1B-7B-0125"
DEVICE = "cuda"
N_SAMPLES = 500  # HellaSwag has ~10K val, PIQA ~1.8K — use 500 for speed

@torch.no_grad()
def eval_hellaswag(model, tokenizer):
    ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
    correct, total = 0, 0
    for item in ds:
        if total >= N_SAMPLES: break
        ctx = item["ctx"]
        endings = item["endings"]
        scores = []
        for end in endings:
            text = ctx + " " + end
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            out = model(**enc, labels=enc["input_ids"])
            scores.append(out.loss.item())
        if scores.index(min(scores)) == int(item["label"]):
            correct += 1
        total += 1
        if total % 100 == 0:
            print(f"  HellaSwag: {total}/{N_SAMPLES} acc={correct/total:.4f}", flush=True)
    return correct / total

@torch.no_grad()
def eval_piqa(model, tokenizer):
    import requests
    base = "https://yonatanbisk.com/piqa/data"
    resp = requests.get(f"{base}/valid.jsonl")
    data = [json.loads(l) for l in resp.text.strip().split("\n") if l.strip()]
    resp_l = requests.get(f"{base}/valid-labels.lst")
    labels = [int(l.strip()) for l in resp_l.text.strip().split("\n") if l.strip()]
    data = data[:N_SAMPLES]; labels = labels[:N_SAMPLES]
    correct, total = 0, 0
    for item, label in zip(data, labels):
        goal, sols = item["goal"], [item["sol1"], item["sol2"]]
        scores = []
        for sol in sols:
            text = goal + " " + sol
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
            out = model(**enc, labels=enc["input_ids"])
            scores.append(out.loss.item())
        if scores.index(min(scores)) == label:
            correct += 1
        total += 1
        if total % 100 == 0:
            print(f"  PIQA: {total}/{len(data)} acc={correct/total:.4f}", flush=True)
    return correct / total

def run_eval(tag, model_path):
    print(f"\n===== {tag} =====", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    hs = eval_hellaswag(model, tokenizer)
    pq = eval_piqa(model, tokenizer)
    print(f"  [{tag}] HellaSwag={hs:.4f} PIQA={pq:.4f}", flush=True)
    return {"hellaswag": hs, "piqa": pq}

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    results = {}
    if mode in ("baseline", "both"):
        results["baseline"] = run_eval("baseline", MODEL_ID)
    if mode in ("lora_merged", "both"):
        if os.path.exists(MERGE_DIR):
            results["beat_lora"] = run_eval("beat_lora", MERGE_DIR)
        else:
            print(f"[skip] merged model not found at {MERGE_DIR}", flush=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] {OUT_PATH}", flush=True)
    for k, v in results.items():
        print(f"  {k}: HellaSwag={v['hellaswag']:.4f} PIQA={v['piqa']:.4f}", flush=True)

if __name__ == "__main__":
    main()
