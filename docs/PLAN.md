# 扩展实验方案：32GB GPU + 128GB 内存机器

本机（30GB 内存）做不了、512GB 纯 CPU 那台没 GPU 加速的几件事，恰好由这台
**32GB 显卡 + 128GB 内存**的机器补齐。6 个实验按论文价值排序，每个对应一个
明确的 limitation 或 mock review 的 6→accept 路径。

## 算力画像与分工

| 资源 | 能做 | 不能做 |
|---|---|---|
| 32GB 显存 | OLMoE-1B-7B（bf16 14GB）、Qwen1.5-MoE bf16 激活 | 真 Mixtral bf16（93GB > 32GB） |
| 128GB 内存 | **Qwen fp32（57GB）** ← 解锁 limitation #3 | Mixtral fp32（187GB） |

三台机器分工：
- **本机（30GB RAM）**：selftest、小模型迭代、论文编辑编译
- **32GB GPU + 128GB RAM（本方案）**：OLMoE 大种子实验 + Qwen fp32 验证 + checkpoint adjudication
- **512GB EPYC 纯 CPU**：真 Mixtral 架构验证（另见 MIXTRAL_RUNBOOK.md）

## 6 个实验与论文价值对应

| # | 实验 | 对应 limitation / 6→accept 路径 | 模型 | 优先级 |
|---|---|---|---|---|
| ① | cost 50 种子排除 Type II | limitation #5；mock review 6→accept 路径 (b) | OLMoE | 最高 |
| ② | Qwen fp32 验证 | limitation #3 内存墙 | Qwen | 高（128GB 独家） |
| ③ | P1 checkpoint adjudication | registered prediction P1 | OLMoE | 中高 |
| ④ | jump_rel anchor recalibration | limitation #3 后半 | Qwen | 中 |
| ⑤ | shared dose-response N 扩样本 | limitation #7（N=176 功效不足） | Qwen | 中 |
| ⑥ | 自然臂 power 扩样本 | limitation（自然臂 E，power pending） | OLMoE/Qwen | 中 |

mock review 两条 6→accept 路径：(a) 真 Mixtral（给 512GB 那台）、(b) cost 50 种子
（本方案 ①）。**两条并行 = 双保险**。

## 文件夹结构

```
expansion_32gb_128gb/
├── docs/
│   └── PLAN.md                  # 本文件（总方案）
├── scripts/
│   ├── exp1_cost_50seeds.py     # ① cost 50 种子
│   ├── exp2_qwen_fp32.py        # ② Qwen fp32 验证
│   ├── exp3_p1_checkpoint.py    # ③ P1 checkpoint adjudication
│   ├── exp4_jump_rel_recalc.py  # ④ jump_rel 重算
│   ├── exp5_dose_response_N.py  # ⑤ N 扩样本
│   └── exp6_natural_arm_power.py# ⑥ 自然臂 power
└── results/                     # 各实验产出 JSON
```

## 通用约定

- **复用而非重写**：所有脚本 `import moe_tear_probe as mtp`（依赖方向：extension ← paper repo），用 sys.path 指向 `moe-tearing-main`。禁止 copy-paste 探针原语。
- **env 驱动**：模型/精度/种子数由环境变量控制，非 argparse（与现有 phase 脚本一致）。
- **路径可移植**：`_HERE = 脚本所在目录` + `MTEAR_PROBE_DIR` 环境变量覆盖，跨机器无需改代码。
- **一次一实验**：避免 OOM。
- **fp32 gate logits**：与 §2 口径一致剥离量化混淆。
- **诚实标注**：每个实验产出 JSON 自带 `evidence_tier` 字段（C/E/I/P），失败按反数据修补铁律报告不重写。

## 环境就绪（这台机器）

```bash
pip install -r moe-tearing-main/requirements.txt   # torch, transformers==5.12.1, numpy<2, datasets, tiktoken
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
nvidia-smi   # 确认 32GB
free -h      # 确认 128GB
```

## 每个实验的现有代码入口与改动量（调研已落地）

| # | 复用脚本/函数 | 复用方式 | 改动点 | 工作量 |
|---|---|---|---|---|
| ① | phase9_experimentD_local.py: `local_soft_gate`(38)/`make_forward`/`patch`/`restore` + phase3 `load_corpus_ids` | import p9，patch 一次跑所有 seed，hard=原生 forward 不 patch | env 化种子数 + paired 窗口采样 + bootstrap CI | 小（~170 行，已验证） |
| ② | phase3 `load_model`(29) + phase3/phase4 `main()` | monkey-patch load_model→device_map=auto+max_memory offload；patch OUT/A_FILE/N_TOKENS | fp32+offload 加载 + Wilcoxon + OLMoE 对照 | 小（~160 行，已验证） |
| ③ | phase3 `main()`（每 checkpoint 跑三控制） | env MTEAR_CKPTS 外层循环，monkey-patch p3.MODEL/load_model | 多 checkpoint 循环 + Δflip@L8 轨迹判读 | 小（~137 行，已验证） |
| ④ | mtp `forward_hard`/`continuity_signature`/`find_boundary_pair`/`discover_moe_blocks`/`capture_hidden_states_multi` | monkey-patch mtp.forward_hard 注入 Qwen shared，continuity_signature 内 modes 元组自动用 patched 版 | forward_hard_shared wrapper ~6 行 + paired 两版对比 | 中（~180 行，已验证） |
| ⑤ | phase3 `main()`（大 N_TOKENS）+ phase4c `main()`（dose-response） | Step1 phase3 大 N 产出 valid rows，Step2 patch phase4c.A_FILE 复用 | N_TOKENS_LARGE env + 曲线稳定性/p 显著性对比 | 中（~150 行，已验证） |
| ⑥ | phase6 `main()`（数据收集）+ analyze_experimentC 偏相关口径 | patch phase6 N_WINDOWS，复现偏相关 + cluster bootstrap by window | 大 N 数据 + cluster bootstrap CI + power 声明 | 中（~160 行，已验证） |

**依赖方向纪律**：6 脚本全部 `import moe_tear_probe as mtp`（extension ← paper repo），
无 copy-paste 探针原语，路径 env 化可移植。全部 py_compile + import 自包含验证通过
（详见 REVIEW.md）。

## 执行顺序建议

1. ①（最高 ROI，纯 OLMoE 32GB GPU，风险最低）
2. ②（128GB 独家优势，解锁 limitation #3）
3. ③（注册预测 P1，single GPU）
4. ④⑤⑥ 视前三项结果再排