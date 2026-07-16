# 剪枝判据信度诊断 — 完整版

> 对应论文附录 `\section{Reliability Diagnostics for Pruning Criteria}` (`paper/sections/08_discussion_appendix.tex` §A)。
> 论文附录只留精炼结论 + tier 标注；本文件留完整实验细节、三轮废弃轮自我撤回、原始数据指向。
> 2026-07-13 解冻写入论文附录。250 窗扩样为 2026-07-11 纯推理 forward（零训练），declared reliability unfreeze，区别于 methodology 的 6 个训练/反事实 forward。

## 概述

one-shot 专家剪枝判据（按单次推理重要性分数排序删尾专家）普遍未报告排序本身的信度。我们在 OLMoE-1B-7B-0125 L8（64 专家，k=8）上加两条诊断：split-half 排序信度 + TypeA 精确反事实因果。结论：剪枝污染因果实在但稀疏（3/64 专家），标准校准集规模下排序大半是噪声（40 窗 SB=0.4201），split-half 信度应作为剪枝论文必报诊断。

## §A.1 split-half 排序信度

- 40 窗标准校准集两独立半，按 one-shot 删除代理打分，raw 半间相关 r = **0.4201**，Spearman-Brown 校正信度 = **0.5916** [E] 0125
- 反解单半每窗相关 r1 ≈ 0.035；信度 0.9 投影需 ~249 窗
- 250 窗扩样实测 r = **0.6666**（校正信度 **0.7999**），仍低于 0.9；实测 r1 ≈ 0.016 比 40 窗估计更小，信度 0.9 所需上修到 ~563 窗，40 窗投影未被证实 [E] 0125
- 结论：标准校准集下 one-shot 排序大半是噪声；split-half 信度为剪枝论文必报诊断 [E]

数据：`results/prune_deconvention_L8_40win_baseline.json`（40 窗基线）、`results/prune_deconvention_L8.json`（250 窗扩样，与 `prune_deconvention_L8_250win.json` 字节相同 md5 `0fde87d6a3158c426dc29771f48f5114`，只留一份）。

## §A.2 TypeA 精确反事实 — 稀疏因果污染

**口径问题**：裸删专家 p 会把删除效应与幸存权重的闭式重归一化（膨胀 1/(1−σ_p)）混在一起。

**TypeA 分离**：A(p) = D − D_restored，D = 含膨胀的删除损伤，D_restored = 把幸存权重恢复到 baseline softmax 值（删 p 召回第 9 + 幸存权重未膨胀）。逐专家因果塌缩，零回归零共线性 [C] 0125。

**结果**：污染因果实在但稀疏 [C]：
- 3/64 专家 A(p) 配对 bootstrap 95% CI 严格 >0（exp16/18/33）
- exp18 铁证：A = +0.00152，CI [+0.00033, +0.00273]，不跨 0
- A/D 中位 17%（非 84%，84% 为 exp18 单例）
- verdict = `RANK_SHIFT_UNEVIDENCED_A_SPARSE_CAUSAL`：剪枝污染因果实在但稀疏，排序位移未被证据支持为全局重排 [C]

数据：`results/prune_deconvention_L8.json`（含 A(p) 配对 bootstrap CI）。

## §A.3 跨层 + 跨模型信度

信度是层特定 + 模型特定 [E]：

| 层/模型 | raw r | SB 校正信度 | verdict |
|---------|-------|-------------|---------|
| L4 (0125) | 0.252 | 0.403 | — |
| L8 40win (0125) | 0.420 | 0.592 | — |
| L12 (0125) | 0.548 | 0.708 | — |
| Qwen1.5-MoE L8 | 0.172 | 0.293 | `A_NEG_ONLY` |

one-shot 剪枝判据不是跨层/跨架构稳定排序；必报诊断建议按层 + 按模型分别报。

数据：`results/prune_deconvention_{L4,L12,L8_Qwen}.json`。

## §A.4 废弃轮完整记录（论文附录只留精炼，此处留全）

三轮每轮自我反杀，记录以便读者审计（反数据修补铁律）：

**轮 1 — 精确版 ρ headline 废弃**：精确版 ρ（含推理修正）作为 headline 被废弃。推理修正后 ρ 仍不构成排序变化证据。

**轮 2 — ρ 尺度混淆废弃**：split-half ρ(D)=0.4201（40 窗两独立半信度，测噪声共享）与配对 ρ(D,D_restored)=0.9322（40 窗两臂共享噪声一致性）是**不同尺度**的两个量。0.93 vs 0.42 任何方向比较不可判读——配对 ρ 本身不构成排序变化证据，0.95 阈值废弃，全局重排无证据 [E]。

> 错误推理已撤回："ρ 高于天花板故证伪"犯了同类尺度错误。

**轮 3 — 0.95 阈值废弃**：0.95 paired 阈值本身废弃。A(p) 配对 bootstrap CI 救出稀疏因果（3/64 显著，exp18 铁证）。

**幸存主张**：§A.2 稀疏因果污染 [C] + §A.1 信度诊断 [E]，均在可辩护证据强度，未重写为更强主张。

## 与论文附录 §A 对应

| 论文附录 §A | 本文件 | 内容差异 |
|-------------|--------|----------|
| §A.1 Split-half reliability | §A.1 | 同；论文带 \tierE |
| §A.2 TypeA exact counterfactual | §A.2 | 同；论文带 \tierC |
| §A.3 Cross-layer and cross-model | §A.3 | 同；论文带 \versiontag + \tierE |
| §A.4 Retracted rounds | §A.4 | 论文精炼 3 句，本文件留完整推理 + 错误撤回 |

## 数据文件清单

- `results/prune_deconvention_L8.json`（250 窗主结果，含 A(p) CI + SB 0.6666）
- `results/prune_deconvention_L8_40win_baseline.json`（40 窗基线，SB 0.4201）
- `results/prune_deconvention_L8_250win.json`（与 L8.json 字节相同，未单列）
- `results/prune_deconvention_L4.json`（SB 0.252）
- `results/prune_deconvention_L12.json`（SB 0.548）
- `results/prune_deconvention_L8_Qwen.json`（SB 0.172，A_NEG_ONLY）