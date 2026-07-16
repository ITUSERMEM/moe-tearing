# 运行手册（32GB GPU + 128GB 内存机器）

6 个扩展实验的执行手册。**铁律：一次一实验，跑完确认 GPU 释放再跑下一个，避免 OOM。**

## 0. 环境就绪（首次）

```bash
# 依赖（与 paper repo 一致）
pip install -r moe-tearing-main/requirements.txt   # torch, transformers==5.12.1, numpy<2, datasets, tiktoken, scipy

# 确认算力
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0))"
nvidia-smi   # 确认 32GB
free -h      # 确认 128GB

# 路径 env（所有实验共用，按实际路径改）
export MTEAR_PROBE_DIR=/home/newtry/moe-tearing-main        # paper repo（moe_tear_probe.py 所在）
export MTEAR_SCRIPTS_DIR=/home/newtry/moe-tear-experiments/scripts  # phase 脚本所在
export MTEAR_OUT=/home/newtry/moe-tear-experiments/expansion_32gb_128gb/results
```

## 1. 执行顺序与命令（按 PLAN.md 优先级）

### ① cost 50 种子（最高 ROI，纯 OLMoE，风险最低）
```bash
cd /home/newtry/moe-tear-experiments/expansion_32gb_128gb/scripts
export MTEAR_MODEL=allenai/OLMoE-1B-7B-0924   # 论文 canonical 0924
python3 exp1_cost_50seeds.py
# 预期 ~20-40 min GPU。产出 exp1_cost_50seeds.json
```

### ② Qwen fp32 验证（128GB 独家，解锁 limitation #3）
```bash
export MTEAR_MODEL=/home/newtry/qwen   # 本地 Qwen1.5-MoE-A2.7B 权重
export N_TOKENS=200                     # 先小 N 验证可行性，跑通后扩 1000
python3 exp2_qwen_fp32.py
# 预期 N=200 约 1-3h（offload 慢），1000 约 5-15h。产出 exp2_qwen_fp32.json
# 跑通后: export N_TOKENS=1000 && python3 exp2_qwen_fp32.py
```

### ③ P1 checkpoint adjudication（注册预测 P1，需早期 checkpoint）
```bash
# 需 OLMoE 早期 checkpoint（HF 公开）。例:
export MTEAR_CKPTS=allenai/OLMoE-1B-7B-0924-ckpt1000,allenai/OLMoE-1B-7B-0924
export MTEAR_CKPT_STEPS=1000,final
export N_TOKENS=300
python3 exp3_p1_checkpoint.py
# 预期 每 checkpoint ~10-20min × N。产出 exp3_p1_checkpoint.json
# 注: checkpoint id 需用户确认 HF 实际可用名（OLMoE 公开 checkpoint 列表）
# 状态: 已完成 (2026-07-10). 三 checkpoint (step5000/500000/final) 全跑完, Δflip@L8 energy 单调增 3.7×
#       (0.0173→0.0511→0.0639), Δdir 平坦内对照, verdict=ACQUIRED, P1 (P)→(E) 训练习得非架构固有.
#       结果 exp3_p1_checkpoint.json + experimentA_ckpt{0,1,2}_*.json, 审查 REVIEW_exp3_p1_acquired.md §五.
#       非悬空实验 (非"中断在 ckpt2/3"): 三 ckpt 数据齐全.
```

### ④ jump_rel 重算（Qwen shared 包含）

### ④ jump_rel 重算（Qwen shared 包含）
```bash
export MTEAR_MODEL=/home/newtry/qwen
export M3_PATHS=50
export SWEEP_LAYERS=8,16
python3 exp4_jump_rel_recalc.py
# 预期 ~15-30 min GPU。产出 exp4_jump_rel_recalc.json
```

### ⑤ N 扩样本 dose-response（Qwen，~2h）
```bash
export MTEAR_MODEL=/home/newtry/qwen
export N_TOKENS_LARGE=5000   # 先 2000 验证再扩 5000
python3 exp5_dose_response_N.py
# 预期 ~2h（phase3 大 N ~37min + phase4c 8 run_b ~1.5h）。产出 exp5_dose_response_N.json
# 状态: 不跑 (废弃, 2026-07-10). exp2 (Qwen fp32) 先证 Qwen bf16 下 Δflip 是量化噪声底
#       (fp32 下 Δflip 塌到 1e-8, cross/near/orth median ~8e-8, 35× gap 系 bf16 量化噪声抬高 Qwen 侧).
#       exp5 要测的剂量响应对象本身即噪声 → 反数据修补铁律下不跑. 详见 REVIEW_exp3_p1_acquired.md §五 line 69.
#       脚本保留 (exp5_dose_response_N.py) 供日后若 Qwen Δflip 被重新证实为真信号时复用.
```

### ⑥ 自然臂 power（OLMoE 0125）
```bash
export MTEAR_MODEL=allenai/OLMoE-1B-7B-0125
export N_WINDOWS_LARGE=400
export N_BOOT=1000
python3 exp6_natural_arm_power.py
# 预期 ~15-30min GPU（400 窗）+ ~2-5min bootstrap。产出 exp6_natural_arm_power.json
```

## 2. 速度阈值与失败回退

- ② Qwen fp32 offload 若 N=200 单 forward >60s → 只跑 N=200 不扩 1000，记录 offload 瓶颈。
- ⑤ 若 phase3 大 N 耗时 >1h → 降 N_TOKENS_LARGE 到 3000，valid rows 仍 ~500（够看曲线）。
- 任何实验 OOM → 确认上一实验 GPU 已释放（`nvidia-smi`），降低 N 参数重跑。
- 路由适配器报错（新 transformers 版本）→ 优先检查 mtp._moe_num_experts/_gate_logits/_expert_forward（CLAUDE.md gotchas）。

## 3. verdict 判读（执行后对照 REVIEW.md tier 映射）

每实验产出 JSON 含 `verdict` + `evidence_tier`。意外结果（REVERSED/INSUFFICIENT_POWER/
MONOTONICITY_BROKEN）按反数据修补铁律报告降级，不重写叙事。

## 4. 打包内容

```
expansion_32gb_128gb/
├── docs/
│   ├── PLAN.md      # 总方案 + 6 实验论文价值对应 + 代码入口调研
│   ├── REVIEW.md    # 审查报告（编译验证 + 复用纪律 + tier 映射）
│   └── RUN.md       # 本文件（运行手册）
├── scripts/
│   ├── exp1_cost_50seeds.py
│   ├── exp2_qwen_fp32.py
│   ├── exp3_p1_checkpoint.py
│   ├── exp4_jump_rel_recalc.py
│   ├── exp5_dose_response_N.py
│   └── exp6_natural_arm_power.py
└── results/         # 各实验产出 JSON（执行后生成）
```