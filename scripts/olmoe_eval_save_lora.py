"""saving OLMoE + LoRA 合并权重到磁盘, 供 lm-eval loading"""
import os, torch
from transformers import AutoModelForCausalLM
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
SAVE_DIR = os.path.join(SCRIPT_DIR, "..", "weights", "olmoe_beat_lora_merged")
MODEL_ID = "allenai/OLMoE-1B-7B-0125"

device = "cuda"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=device, torch_dtype=torch.bfloat16)

lora_path = os.path.join(OUT_DIR, "olmoe_beat_lora_lora_l1_0_lora.pt")
sd = torch.load(lora_path, map_location=device)
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

model.save_pretrained(SAVE_DIR, safe_serialization=True)
print(f"[save] merged model -> {SAVE_DIR}", flush=True)
