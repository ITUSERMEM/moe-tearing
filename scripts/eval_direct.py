"""eval_direct.py — 直接loading合并模型跑 HellaSwag / MMLU / GSM8K

用法: python3 scripts/eval_direct.py baseline|beat
"""

import os, sys, json, torch, torch.nn.functional as F, re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "..", "weights")
DEVICE = "cuda"

def load_model(tag):
    if tag == "baseline":
        path = os.path.join(WEIGHTS_DIR, "olmoe_beat_lora_baseline_merged")
    elif tag == "beat":
        path = os.path.join(WEIGHTS_DIR, "olmoe_beat_lora_0.05_merged")
    else:
        raise ValueError(tag)
    print(f"[load] {tag} from {path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(path, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

@torch.no_grad()
def eval_hellaswag(model, tokenizer, n=500):
    ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
    correct, total = 0, 0
    for item in ds:
        if total >= n: break
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
            print(f"  HellaSwag: {total}/{n} acc={correct/total:.4f}", flush=True)
    return correct / total

@torch.no_grad()
def eval_mmlu(model, tokenizer, n=200):
    ds = load_dataset("mmlu", "all", split="test", streaming=True)
    correct, total = 0, 0
    for item in ds:
        if total >= n: break
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]
        letters = ["A", "B", "C", "D"]
        scores = []
        for i, choice in enumerate(choices):
            text = f"{question} {letters[i]}. {choice}"
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
            out = model(**enc, labels=enc["input_ids"])
            scores.append(out.loss.item())
        if scores.index(min(scores)) == answer:
            correct += 1
        total += 1
        if total % 50 == 0:
            print(f"  MMLU: {total}/{n} acc={correct/total:.4f}", flush=True)
    return correct / total

@torch.no_grad()
def eval_gsm8k(model, tokenizer, n=200):
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    correct, total = 0, 0
    for item in ds:
        if total >= n: break
        question = item["question"]
        # Extract answer from the answer field: "#### 42"
        answer_match = re.search(r'####\s*(-?\d+\.?\d*)', item["answer"])
        if not answer_match: continue
        correct_answer = answer_match.group(1)
        # Generate
        enc = tokenizer(question, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        out = model.generate(**enc, max_new_tokens=128, temperature=0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        # Extract number from generation
        pred_match = re.search(r'(-?\d+\.?\d*)', gen.replace(",", ""))
        if pred_match and pred_match.group(1) == correct_answer:
            correct += 1
        total += 1
        if total % 50 == 0:
            print(f"  GSM8K: {total}/{n} acc={correct/total:.4f}", flush=True)
    return correct / total

def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if tag not in ("baseline", "beat"):
        print("Usage: python3 eval_direct.py baseline|beat")
        return
    out_label = "beat_0.05" if tag == "beat" else "baseline"
    model, tokenizer = load_model(tag)
    results = {}
    for name, fn, n in [("hellaswag", eval_hellaswag, 500),
                          ("mmlu", eval_mmlu, 200),
                          ("gsm8k", eval_gsm8k, 200)]:
        print(f"\n[{tag}] running {name} (n={n})...", flush=True)
        acc = fn(model, tokenizer, n=n)
        results[name] = acc
        print(f"  [{tag}] {name}: {acc:.4f}", flush=True)
    out_path = os.path.join(OUT_DIR, f"benchmark_{out_label}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] {out_path}", flush=True)

if __name__ == "__main__":
    main()
