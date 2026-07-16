"""olmoe_eval_downstream.py — OLMoE BEAT LoRA 下游任务评测

loading OLMoE-0125 + LoRA 权重, 测试 HellaSwag + PIQA.
output: results/olmoe_eval_downstream.json
"""

import os, sys, json, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
OUT_PATH = os.path.join(OUT_DIR, "olmoe_eval_downstream.json")
MODEL_ID = "allenai/OLMoE-1B-7B-0125"
DEVICE = "cuda"

def merge_lora(model, lora_path):
    sd = torch.load(lora_path, map_location=DEVICE)
    moe_blocks = [(n,m) for n,m in model.named_modules() if isinstance(m, OlmoeSparseMoeBlock)]
    for li, (_, blk) in enumerate(moe_blocks):
        ag = sd[f"L{li}_A_gate"].to(DEVICE).float()
        bg = sd[f"L{li}_B_gate"].to(DEVICE).float()
        ad = sd[f"L{li}_A_down"].to(DEVICE).float()
        bd = sd[f"L{li}_B_down"].to(DEVICE).float()
        gu = blk.experts.gate_up_proj.data.float()
        dn = blk.experts.down_proj.data.float()
        for e in range(gu.shape[0]):
            gu[e] = gu[e] + (ag[e] @ bg[e]).to(gu.dtype)
            dn[e] = dn[e] + (ad[e] @ bd[e]).to(dn.dtype)
        blk.experts.gate_up_proj.data = gu.to(torch.bfloat16)
        blk.experts.down_proj.data = dn.to(torch.bfloat16)
    return model

@torch.no_grad()
def eval_hellaswag(model, tokenizer, n_samples=500):
    ds = load_dataset("hellaswag", split=f"validation[:{n_samples}]")
    correct, total = 0, 0
    for item in ds:
        ctx = item["ctx"]
        endings = [item[f"ending{i}"] for i in range(4)]
        scores = []
        for end in endings:
            text = ctx + " " + end
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            out = model(**enc, labels=enc["input_ids"])
            scores.append(out.loss.item())
        if scores.index(min(scores)) == int(item["label"]):
            correct += 1
        total += 1
    return correct / total

@torch.no_grad()
def eval_piqa(model, tokenizer, n_samples=500):
    ds = load_dataset("piqa", split=f"validation[:{n_samples}]")
    correct, total = 0, 0
    for item in ds:
        goal = item["goal"]
        sols = [item["sol1"], item["sol2"]]
        scores = []
        for sol in sols:
            text = sol if sol.startswith(goal[:10]) else goal + " " + sol
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
            out = model(**enc, labels=enc["input_ids"])
            scores.append(out.loss.item())
        if scores.index(min(scores)) == int(item["label"]):
            correct += 1
        total += 1
    return correct / total

def main():
    results = {}

    # 原模型
    print("[eval] loading baseline...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    for tag in ["baseline", "beat_lora"]:
        if tag == "beat_lora":
            lora_path = os.path.join(OUT_DIR, "olmoe_beat_lora_lora_l1_0_lora.pt")
            if not os.path.exists(lora_path):
                print(f"[eval] LoRA weights not found at {lora_path}", flush=True)
                continue
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE, torch_dtype=torch.bfloat16)
            model = merge_lora(model, lora_path)
            model.eval()
            print("[eval] LoRA merged, evaluating...", flush=True)

        print(f"[eval] {tag} HellaSwag...", flush=True)
        hs = eval_hellaswag(model, tokenizer, n_samples=300)
        print(f"  HellaSwag: {hs:.4f}", flush=True)

        print(f"[eval] {tag} PIQA...", flush=True)
        pq = eval_piqa(model, tokenizer, n_samples=300)
        print(f"  PIQA: {pq:.4f}", flush=True)

        results[tag] = {"hellaswag": hs, "piqa": pq}

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] {OUT_PATH}", flush=True)

if __name__ == "__main__":
    main()
