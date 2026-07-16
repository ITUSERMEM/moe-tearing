"""patch_json_train_loss.py — loading已存最终权重, 重测训练集 loss, 补丁 JSON

用法: python3 scripts/patch_json_train_loss.py
"""

import os, sys, json, torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, GraniteMoeForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "..", "weights")
DATA_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "datasets--wikitext", "snapshots", "b08601e04326c79dfdd32d625aee71d232d685c3",
    "wikitext-103-raw-v1")
MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-base"
PAMS_WIN = 128
PAMS_STEP = 64
PAMS_BS = 4
VAL_OFFSET = 600000
TRAIN_HOLDOUT_OFFSET = 500000

def load_wikitext103_tokens(offset=0, max_tokens=None):
    import pandas as pd
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = []
    for fn in ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]:
        fp = os.path.join(DATA_CACHE, fn)
        df = pd.read_parquet(fp)
        for text in df["text"]:
            if isinstance(text, str) and len(text.strip()) > 0:
                texts.append(text)
    all_ids = []
    for i in range(0, len(texts), 1000):
        batch = texts[i:i+1000]
        enc = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for ids in enc: all_ids.extend(ids)
    ids = torch.tensor(all_ids, dtype=torch.long)
    if offset > 0:
        ids = ids[offset:]
    if max_tokens is not None:
        ids = ids[:max_tokens]
    return ids

def measure_loss(model, ids, n_win=50, device="cuda"):
    model.eval()
    losses = []
    with torch.no_grad():
        for wi in range(0, n_win, PAMS_BS):
            n = min(PAMS_BS, n_win - wi)
            wids = torch.stack([ids[(wi+j)*PAMS_STEP:(wi+j)*PAMS_STEP+PAMS_WIN].to(device) for j in range(n)])
            out = model(**dict(input_ids=wids, attention_mask=torch.ones_like(wids)))
            l = F.cross_entropy(out.logits[:,:-1].float().reshape(-1, out.logits.size(-1)), wids[:,1:].reshape(-1), reduction="none")
            losses.append(l.mean().item())
    model.train()
    return float(np.mean(losses))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[patch] device={device}", flush=True)

    ids_train = load_wikitext103_tokens(offset=TRAIN_HOLDOUT_OFFSET, max_tokens=PAMS_WIN*50)
    ids_val = load_wikitext103_tokens(offset=VAL_OFFSET, max_tokens=PAMS_WIN*50)
    print(f"[patch] train_eval_tokens={len(ids_train)} val_tokens={len(ids_val)}", flush=True)

    for tag in ["l0_0", "l0_1", "l1_0"]:
        json_path = os.path.join(OUT_DIR, f"granite_pams_align_{tag}.json")
        wt_path = os.path.join(WEIGHTS_DIR, f"granite_pams_align_{tag}")

        if not os.path.exists(json_path):
            print(f"[patch] {tag}: JSON not found, skip", flush=True)
            continue
        if not os.path.exists(wt_path):
            print(f"[patch] {tag}: weights not found at {wt_path}, skip", flush=True)
            continue

        print(f"\n[patch] {tag}: loading weights from {wt_path}", flush=True)
        model = GraniteMoeForCausalLM.from_pretrained(wt_path, device_map=device, torch_dtype=torch.bfloat16)

        train_loss = measure_loss(model, ids_train, n_win=50, device=device)
        val_loss = measure_loss(model, ids_val, n_win=50, device=device)
        print(f"[patch] {tag}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}", flush=True)

        with open(json_path) as f:
            d = json.load(f)

        d["patch_note"] = "loss=train_CE, val_loss=validation_CE (patched)"
        d["before"]["loss"] = d["before"].get("loss", d["before"]["val_loss"])
        d["before"]["val_loss"] = d["before"]["val_loss"]
        d["after"]["loss"] = train_loss
        d["after"]["val_loss"] = val_loss

        with open(json_path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"[patch] {tag}: patched OK", flush=True)

        del model
        torch.cuda.empty_cache()

    print("\n[patch] All done", flush=True)

if __name__ == "__main__":
    main()
