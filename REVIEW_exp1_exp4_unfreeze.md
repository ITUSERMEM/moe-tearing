# 审查报告: ①④ verdict 回写 + 解冻范围声明

日期: 2026-07-10
范围: ① 50seed cost 零结果回写 + ④ jump_rel shared-inclusive verdict 回写 + 解冻声明入论文 + 编译验证

## 背景
盘点指出扩展包 4 个 verdict 尚未回写论文 limitation 段。核实: ②③⑥ 已在前轮回写; 真实缺口为 ①(50seed cost) 和 ④(jump_rel)。另: 扩展包 6 实验是新 forward, 与 07-09 主线冻结声明形式冲突, 需补解冻范围声明。

## 一. ① 50seed cost 零结果回写 (ZERO_RESULT_AT_POWER, E)

### 数据 (exp1_cost_50seeds.json, OLMoE-1B-7B-0125 bf16 k=8 m0=0.05)
- 50-seed paired bootstrap CI: median -0.00162, 95%CI[-0.00512,+0.00278] (含零), n_pos/n_neg=24/26
- verdict=ZERO_RESULT_AT_POWER, tier E

### 反数据修补铁律约束 (关键)
① 口径 = per-seed median loss 差 (OLMoE-0125 local_soft), 与论文 05 phase9 两跑 (+0.40%/-1.15%, 不同窗口聚合) **不同口径, 不可直接比较**。① 零结果**不重写**为支持/反对 phase9 两跑, 仅作**独立口径佐证零结果**追加, 显式标注不可比较。

### 论文变更 (2 处)
1. 05_governance_law sec:law_positive cost 段: 追加"50-seed paired-bootstrap re-run on distinct OLMoE-1B-7B-0125 config (not directly comparable to phase-9 two-run) reproduces zero result, median -0.00162, 95%CI[-0.00512,+0.00278] includes zero, 24/26 pos/neg seeds (E)"
2. 08_discussion_appendix limitation #5: 追加同 50seed 数字 + 显式标注"a distinct config from the phase-9 two-run above, not directly comparable...consistent with the two-run zero result above without overriding it (E)"
3. 01_intro: 高层保持 ~3% power 表述(不加具体数字, 避免重复)

### 不改的
- phase9 两跑表述 (+0.40%/-1.15%, paired bootstrap [-0.00003,0.00014]) 保留不动, ① 是追加佐证非替换
- 不把 ① 50seed 当作对 phase9 的"修正"或"否定"

## 二. ④ jump_rel shared-inclusive verdict 回写 (SHARED_INFLATES_DENOMINATOR, E)

### 数据 (exp4_jump_rel_recalc.json, Qwen bf16 k=4, 50 paths paired)
- mean_drop_frac = 0.5692 (shared-inclusive jump_rel 比 shared-exclusive routed-only 低 ~57%)
- verdict=SHARED_INFLATES_DENOMINATOR, tier E
- 机制: Qwen always-active shared expert 经 sigmoid(shared_gate) 连续叠加进 block output 分母, shared-inclusive median||y|| 更大, jump_rel=jump_abs/median||y|| 更低

### ② 的口径张力 (关键区分)
④ 用 Qwen bf16, ② 揭示 Qwen bf16 的 **KL Δflip** 是噪声底(~1e-5)。但 ④ 的 jump_rel 是**范数口径**(block-output norm ~0.37 大尺度), 量级远超 ② 的 KL 噪声底 4 个数量级, **不受 ② 污染**。论文 06/08 显式标注"norm-ratio caliber ~0.37, well above ~1e-5 KL noise floor, not contaminated by it"。

### 论文变更 (2 处)
1. 06_anchor_reckoning caliber clarification 段: "true shared-inclusive value is lower" 追加 50-path paired 实测 drop_frac=0.569 (~57% lower) (E) + 范数口径不受②KL噪声底污染 caveat
2. 08_discussion_appendix limitation #3 后半: 原 (P)"needs sweep-scale forward...cannot recomputed...tag (P)" 拆为:
   - paired 50-path 实测 (E, drop_frac=0.569, 范数口径不受②污染)
   - full sweep-scale 仍 (P) (sweep_qwen.json 只存 scalar summaries 无法从 JSON 重算)

### 不升 full (C)
④ 是 paired 50-path sample, 非完整 sweep-scale (every token every layer)。诚实拆分 paired(E) + sweep(P)。

## 三. 解冻范围声明 (08 methodology)

扩展包 6 实验是新 forward, 与 07-09 主线冻结声明形式冲突。在 08_discussion_appendix methodology 段 "eight experiments" 句后追加解冻声明:
- 主线 2026-07-09 冻结(无新 forward, 仅基于已 commit JSON 分析)
- 6 个扩展 forward 在 32GB-GPU/128GB-RAM 节点跑, 各有预注册 bet + end-table: ① 50seed cost / ② Qwen fp32 / ③ P1 checkpoint / ④ jump_rel / ⑤ dose-response N(未跑, Qwen bf16 是噪声底) / ⑥ natural-arm power
- verdicts 按 no-data-patching 铁律回写相应 limitation/registered-prediction 段(证伪的降级, 不重写为支持不同主张)

## 四. 验证
1. 编译 pdflatex×3 + bibtex: 0 errors / 0 [?] / 0 undefined / 0 overfull / 0 LaTeX Warning. pdf 715279 bytes (从 711469 +3.8KB, ①④+解冻声明合理)。
2. 数字核对: 0.569 / 57% (④ drop_frac), -0.00162 / [-0.00512,+0.00278] / 24/26 (① 50seed), 均与 JSON 一致。
3. tier 审计: ① 追加 (E), ④ 拆 paired(E)+sweep(P), 解冻声明引各 \cref。
4. 反数据修补: ① 不覆盖 phase9(独立口径佐证), ④ 不与②冲突(范数 vs KL 口径区分), ⑤ 不跑如实记录(非零结果掩盖)。
5. \versiontag: ① 0125, ④ Qwen (无 versiontag, 跨模型), 解冻声明 0125+0925+0924 各引。

## 五. 战役完整收尾 (①-⑥ 全部 verdict 已回写论文/记忆)
| 实验 | verdict | tier | 论文回写位置 |
|---|---|---|---|
| ① | ZERO_RESULT_AT_POWER | E | 05 sec:law_positive + 08 lim#5 (50seed佐证,不覆盖phase9) |
| ② | FP32_NOISE_FLOOR_GAP_INFLATED_BF16 | E | 07候选1/35×/60×/reverse-falsify + 08 lim#3前半 + 01贡献5 (全链路降级) |
| ③ | ACQUIRED | E | 08 P1段 + 08 lim#1 + 07 What-this-means (P1(P)->(E)) |
| ④ | SHARED_INFLATES_DENOMINATOR | E | 06 anchor + 08 lim#3后半 (paired(E)+sweep(P)) |
| ⑤ | 不跑(②已覆盖) | - | 08 methodology解冻声明如实记录"not run" |
| ⑥ | POWERFUL_NULL | E | 03 sec:dyn_natural + 01 (power declared) |
| 解冻声明 | - | - | 08 methodology (主线07-09冻结 + 6扩展forward解冻范围) |

编译均干净(0 err/0[?]/0 overfull)。所有 verdict 按 no-data-patching 铁律回写, 不改主线叙事。

链接: [[exp1-cost-50seeds-findings]], [[exp2-qwen-fp32-findings]], [[exp3-p1-acquired-findings]], [[exp4-jump-rel-shared-inflates-findings]], [[exp6-natural-arm-power-findings]], [[icml-paper-external-review]]。