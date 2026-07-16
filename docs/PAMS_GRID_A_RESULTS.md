# 格A结果：PAMS on Granite-1B from-scratch — 明确负结果

## 总表 (seed=42, 5000步, Wikitext-103)

| λ | loss | flip_avg | dt_avg | flip vs λ=0 | dt vs λ=0 |
|---|------|---------|--------|------------|----------|
| 0 (hard) | 6.48 | **0.0455** | **0.130** | — | — |
| 1.0 | 6.56 | 0.0728 | 3.078 | ≈持平(+60%) | **23×膨胀** |
| 10.0 | 6.67 | 0.0822 | 2.796 | ≈持平(+81%) | **22×膨胀** |
| 100.0 | 6.76 | 0.0629 | 2.634 | ≈持平(+38%) | **20×膨胀** |

## 核心发现

**PAMS 成功推宽 d_t（20-23×），但完全不影响 flip 率。**

所有 λ (1, 10, 100) 下 flip 率稳定在 4.5-8.2%，与 λ=0 基线无统计学差异。
d_t 中位从 0.13（基线）升到 2.6-3.1（PAMS），提升 20-23×。

## 机制解释

PAMS 的目标量 d_t = z[k-1] - z[k]（第 k 大与第 k+1 大 logit 的 gap）。

但 flip 的原因不是单一 (k, k+1) 边界不稳，而是：
1. **内部 top-k 重排序**：k=8 选了 8 个专家，其中任意两个的排序在 bf16 vs fp32 下都可能反转
2. **多边界问题**：top-8 内有 7 个内部边界 + 1 个退出边界。PAMS 只推退出边界 d_t
3. **随机路由噪声**：随机初始化下 gate weight 噪声大，排序不稳定

## 对 PAMS 的最终 verdict

**PAMS 不是有用的方法。** 在三设置（probe OK → 真模型 gate-only 失败 → 真模型全参失败 → Granite from-scratch 失败）均确认。

关键教训：d_t 推宽不降 flip，因为 d_t 不是 flip 的因果机制。

## 文件

| 文件 | λ | 大小 |
|------|---|------|
| `results/granite_pams_l0_0.json` | 0 (hard) | 21KB |
| `results/granite_pams_l1_0.json` | 1.0 | 21KB |
| `results/granite_pams_l10_0.json` | 10.0 | 21KB |
| `results/granite_pams_l100_0.json` | 100.0 | 21KB |
