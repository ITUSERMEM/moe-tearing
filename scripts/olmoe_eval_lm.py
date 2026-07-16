"""olmoe_eval_lm.py — OLMoE BEAT LoRA lm-eval-harness 评测

用法: python3 scripts/olmoe_eval_lm.py           # baseline
      python3 scripts/olmoe_eval_lm.py --lora     # loading LoRA 权重复测
"""

import os, sys, json, torch, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
OUT_PATH = os.path.join(OUT_DIR, "olmoe_eval_lm.json")
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", action="store_true")
    args = parser.parse_args()

    tag = "beat_lora" if args.lora else "baseline"
    print(f"[eval] {tag}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE, torch_dtype=torch.bfloat16)

    if args.lora:
        lora_path = os.path.join(OUT_DIR, "olmoe_beat_lora_lora_l1_0_lora.pt")
        if not os.path.exists(lora_path):
            print(f"[eval] LoRA not found at {lora_path}", flush=True)
            return 1
        merge_lora(model, lora_path)
        print("[eval] LoRA merged", flush=True)

    model.eval()

    import lm_eval
    tasks = ["hellaswag", "piqa"]
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={MODEL_ID},dtype=bfloat16",
        tasks=tasks,
        num_fewshot=0,
        batch_size=4,
        device=DEVICE,
    )

    out = {tag: {}}
    for task in tasks:
        acc = results["results"][task].get("acc,none", results["results"][task].get("acc"))
        out[tag][task] = acc
        print(f"  {task}: {acc:.4f}", flush=True)

    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            existing = json.load(f)
        existing.update(out)
    else:
        existing = out
    with open(OUT_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[done] {OUT_PATH}", flush=True)

if __name__ == "__main__":
    main()
