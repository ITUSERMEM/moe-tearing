# MoE Routing Tear: Continuity Diagnostics and Governance

This repository contains the experimental code accompanying the paper:

**"A Denominator-Invariance Condition for Mixture-of-Experts Routing: Post-Hoc Constraints, Training-Time Failures, and a Constructive Dual"**

## Structure

```
scripts/
├── granite_pams_align.py          # BEAT: Boundary-Expert Alignment Training (main experiment)
├── granite_pams_from_scratch.py   # PAMS margin-pushing from scratch
├── granite_pams_rel.py            # Relative margin PAMS
├── real_pams_finetune.py          # Real-model PAMS fine-tuning (gate only)
├── real_pams_adafactor.py         # Full-parameter PAMS with Adafactor
├── real_pams_gate_expert.py       # Gate + expert PAMS fine-tuning
├── real_pams_eps_calib.py         # Epsilon calibration for PAMS
├── olmoe_beat_ft.py              # OLMoE BEAT fine-tuning (target layer)
├── olmoe_beat_ft_all.py          # OLMoE BEAT fine-tuning (all layers)
├── olmoe_beat_lora.py            # OLMoE BEAT + LoRA
├── prune_deconvention.py          # Pruning criteria reliability diagnostics
├── jaccard_analysis.py            # Cross-arm routing Jaccard similarity
├── expert_swap_demo.py           # Expert replacement jump demonstration
├── eval_direct.py                 # Downstream task evaluation (HellaSwag, etc.)
├── c4_*.py                        # C4-based collateral experiments
├── exp*.py                        # Extension experiments (A-F)
└── run_*.py                       # Batch run scripts
```

## Key Experiments

### BEAT (Boundary-Expert Alignment Training)
```bash
# Granite from-scratch, λ=1.0, 20k steps
PAMS_LAMBDA=1.0 PAMS_STEPS=20000 PAMS_TAG=l1_0 python3 scripts/granite_pams_align.py

# Multi-seed replication
python3 scripts/run_multiseed.py
```

### PAMS Margin-Pushing
```bash
# Gate-only fine-tuning (OLMoE)
PAMS_LAMBDA=0.05 PAMS_EPS=0.0116 python3 scripts/real_pams_finetune.py
```

### Cross-Model Verification
```bash
# Mixtral cross-convention verification (CPU, fp32)
python3 moe_tear_probe.py --sweep --model mistralai/Mixtral-8x7B-v0.1 --device cpu
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- transformers ≥ 4.38
- datasets, numpy (< 2.0)

## Citation

```
@inproceedings{denominator2026,
  title={A Denominator-Invariance Condition for Mixture-of-Experts Routing},
  author={Anonymous},
  booktitle={ICML 2026},
  year={2026}
}
```

## License

MIT
