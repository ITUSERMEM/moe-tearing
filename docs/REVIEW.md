# 扩展实验审查报告（32GB GPU + 128GB 内存机器）

本报告记录 6 个扩展实验的方案、代码、复用点、验证结果与对论文的影响。
按 CLAUDE.md 全局约束，每次完成修改任务后必须输出审查报告并保存为 .md。

## 实验清单与状态

| # | 实验 | 脚本 | 复用入口 | 状态 | 对论文影响 |
|---|---|---|---|---|---|
| ① | cost 50 种子 | exp1_cost_50seeds.py | phase9 local_soft_gate/make_forward/patch/restore + phase3 load_corpus_ids；hard 基线=原生 forward 不 patch（与 ph9 line 138 一致） | 代码完成+编译验证通过,待跑 | limitation #5；6→accept 路径 (b) |
| ② | Qwen fp32 验证 | exp2_qwen_fp32.py | phase3 load_model（monkey-patch device_map=auto+max_memory offload）+ phase3/phase4 main()；OLMoE 对照用 phase4d 硬编码 DFLIP_OLMOE | 代码完成+编译验证通过,待跑 | limitation #3 内存墙（fp32 57GB） |
| ③ | P1 checkpoint | exp3_p1_checkpoint.py | phase3 main()（每 checkpoint 跑三控制算 Δflip@L8=median(cross-near) energy）；env 驱动多 checkpoint | 代码完成+编译验证通过,待跑 | registered P1（acquired vs intrinsic） |
| ④ | jump_rel 重算 | exp4_jump_rel_recalc.py | mtp.forward_hard/continuity_signature/find_boundary_pair/discover_moe_blocks/capture_hidden_states_multi/SWEEP_PROBE_TEXTS；monkey-patch forward_hard 注入 Qwen shared | 代码完成+编译验证通过,待跑 | limitation #3 后半（jump_rel 分母 ||y|| 不含 shared） |
| ⑤ | N 扩样本 | exp5_dose_response_N.py | phase3 main()（大 N_TOKENS 产出 valid rows）+ phase4c main()（dose-response）；对比原 experimentB_shared_ablation_qwen.json (N=176) | 代码完成+编译验证通过,待跑 | limitation #7（N=176 功效不足） |
| ⑥ | 自然臂 power | exp6_natural_arm_power.py | phase6 main()（大 N_WINDOWS 数据收集）+ analyze_experimentC 偏相关口径（控制 freq+norm+position）；cluster bootstrap by window | 代码完成+编译验证通过,待跑 | 自然臂 E→E+power 声明（r≈0.02 零结果） |

## 复用纪律验证

- [x] 所有脚本 `import moe_tear_probe as mtp`，依赖方向 extension ← paper repo（无反向导入）
- [x] 未 copy-paste 探针原语（forward_hard/_combine/discover_moe_blocks/continuity_signature 等均调 mtp/phase 函数）
- [x] 路径可移植（`_HERE` + `MTEAR_PROBE_DIR`/`MTEAR_SCRIPTS_DIR` env 覆盖），跨机器无需改代码
- [x] env 驱动（MTEAR_MODEL/K/N_TOKENS/N_WINDOWS 等），非 argparse（与现有 phase 脚本一致）
- [x] 产出 JSON 自带 `evidence_tier` 字段（全部标 E，执行后按 verdict 升级）
- [x] py_compile + import 自包含验证通过（见下）

## 编译验证

```
py_compile (6/6 通过):
  OK  exp1_cost_50seeds.py
  OK  exp2_qwen_fp32.py
  OK  exp3_p1_checkpoint.py
  OK  exp4_jump_rel_recalc.py
  OK  exp5_dose_response_N.py
  OK  exp6_natural_arm_power.py

复用点存在性 (mtp 原语 + phase 入口, 全 OK):
  mtp: forward_hard / continuity_signature / find_boundary_pair /
       discover_moe_blocks / capture_hidden_states_multi / set_norm_topk /
       SWEEP_PROBE_TEXTS / run_sweep
  phase: phase3_experimentA / phase4_experimentB / phase4c_qwen_shared_ablation /
         phase9_experimentD_local / phase6_experimentC

import 自包含 (MTEAR_PROBE_DIR 指向 moe-tearing-main, 6/6 通过):
  OK exp1..exp6 全部 import 成功（不触发 main, 不加载模型）

关键签名核对:
  continuity_signature(block, h_a, h_b, resolutions, k, tau) -> exp4 调用匹配
  find_boundary_pair(mode=, generator=, return_meta=) -> exp4 调用匹配
  forward_hard 返回 (out, mask) tuple -> exp4 forward_hard_shared 包装匹配
  exp4 无 shared_expert 时 hasattr 守卫退化为原生（OLMoE 路径安全）
```

## 对论文 tier 的影响（执行后填充）

（各实验 verdict 回报后填充 §5/§8/limitation 的具体 tier 升级。预期映射：
- ① ZERO_RESULT_AT_POWER → limitation #5 从 2-run 零结果升 50-seed 零结果；
  DETECTED → 零结果升有功效测量
- ② FP32_VALIDATED → limitation #3 fp32 验证落地，堵 C' bf16 噪声底质询；
  FP32_REVERSED → 铁律报告降级
- ③ INTRINSIC → P1 falsified acquired（P→E）；ACQUIRED → P1 confirmed（P→C/E）
- ④ SHARED_INFLATES_DENOMINATOR → limitation #3 后半量化 jump_rel 锚定偏差
- ⑤ STABLE_AT_LARGE_N → limitation #7 功效升级；MONOTONICITY_BROKEN → 铁律降级
- ⑥ POWERFUL_NULL → 自然臂 E→E+power；INSUFFICIENT_POWER → 报告降级）

## 反数据修补铁律遵守

若任一实验结果意外（方向反转/量级不符），按铁律报告降级，不重写叙事。
每个脚本 verdict 分支均含 REVERSED/INSUFFICIENT_POWER/MONOTONICITY_BROKEN 等意外路径，
执行后在此记录实际意外结果及降级处理。