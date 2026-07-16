# 7.1 约定感知剪枝判据去污染 — 后续实验清单

日期: 2026-07-11
状态: P-A~C 已完成, P-D 需 512GB Windows 机
脚本: `scripts/prune_deconvention.py`
冻结纪律: 纯推理零训练

## 实验结果 (2026-07-11)

### P-A: L8 ≥250窗扩窗 ✅ DONE

命令:
```bash
PD_LAYER=8 PD_N_WIN=250 python3 prune_deconvention.py
```

实际结果:
```
split-half ρ(D): 0.5916 (40w) → 0.6666 (250w)
SB250: 0.7999 (未达0.9)
单窗r₁: 0.03495(40w高估) → 0.01574(250w修正)  ← 关键修正
0.9需窗: 248.5(40w) → 563(250w)               ← 反推值太高
A显著正: 3/64 → 4/64 (边际+1)
A/D中位: 0.17 → 0.071
exp18 A CI: [+0.00033,+0.00273] → [-0.00041,+0.00082] 跨0  ← 个体案例退火
```

verdict: 方向确认但量级修正. r₁ 被 40 窗高估 2×. A 稀疏因果维持 (C). 反数据修补如实报告.

### P-B: L4/L12 多层 ✅ DONE

命令:
```bash
PD_LAYER=4  PD_N_WIN=40  # → L4.json
PD_LAYER=12 PD_N_WIN=40  # → L12.json
```

实际结果:
| 层 | ρ(D) | SB40 | r₁ | A显著正 | A/D中位 | rank_shift_max |
|----|------|------|----|---------|---------|----------------|
| L4 | 0.2523 | 0.4030 | 0.0166 | 4/64 | -0.023 | 35 |
| L8 (250w) | 0.6666 | 0.7999 | 0.0157 | 4/64 | 0.071 | 16 |
| L12 | 0.5484 | 0.7083 | **0.0572** | **17/64** | **0.265** | **58** |

verdict: 层异质性显著 (E). 晚层 L12 塌缩伪影更强 (17/64 显著正, AD 占比 26.5%). 早层 L4 信度极低, 排序不可靠. 如实报, 不强行单调.

### P-C: 跨模型 Qwen 一臂 ✅ DONE

命令:
```bash
MTEAR_MODEL=/home/newtry/qwen MTEAR_K=4 PD_LAYER=8 PD_N_WIN=40
# ⚠️ 输出 L8.json → 立即改名 L8_Qwen.json (脚本不含模型名)
```

实际结果:
```
ρ(D) = 0.1718 (极低)
A显著正 = 0/60  A显著负 = 14/60  ← 方向与OLMoE相反!
A/D中位 = -0.747 (塌缩使损伤被系统性低估)
r₁ = 0.0103/窗 (更低), 0.9需867窗
峰值显存: 30.41GB (接近极限30.5GB)
```

verdict: 信度低普适 (E 强化). A(p) 方向跨模型相反 — 塌缩伪影存在性普适但方向非普适.

### P-D: Mixtral 🔴 未跑

阻塞: 模型仅 512GB Windows 机 (D:\htlllm\Mixtral-8x7B-v0.1, fp32 187GB).
本机 32GB GPU 不可加载. 另需适配 Mixtral 路由 (k=2, 8专家/层, norm_topk_prob=True).

## 输出路径 bug (已修复记录)

脚本 `prune_deconvention.py:L516`:
```python
outpath = os.path.join(OUT, f"prune_deconvention_L{PD_LAYER}.json")
```

**不含模型名**. 跨模型运行即覆盖. 修复方案 (待改):
```python
model_short = MODEL.split("/")[-1].replace("-", "_")  # OLMoE_1B_7B_0125 / qwen
outpath = os.path.join(OUT, f"prune_deconvention_{model_short}_L{PD_LAYER}.json")
```

## 论文策略

### 主结果 tier C
- A(p) 稀疏因果: OLMoE L8 4/64 显著正 + L12 17/64 显著正 (因果反事实, 两臂唯一差异=膨胀因子)
- 单专家 A/D 最高 ~84% (40w exp18), 但 250w 跨0 → 用中位/A分布而非个体案例

### 副结果 tier E
- split-half 信度诊断普适低: OLMoE ρ=0.25-0.67, Qwen ρ=0.17
- r₁ ≈ 0.01-0.06/窗 (模型/层依赖)
- 主张: 剪枝判据文献应报 split-half 信度 (打 REAP / 统一框架 / EAC-MoE)

### 附录三轮反杀
1. ρ(D,D_restored)<1 非排序变化证据 (0.95 阈值废弃)
2. split-half 信度天花板: 40窗不可信, 250窗ρ=0.80仍不足0.9
3. r₁ 修正 (0.035→0.0157) + exp18 退化 — 诚实预先卸除攻击

## 文件

| 文件 | 内容 | 大小 |
|------|------|------|
| `results/prune_deconvention_L8_40win_baseline.json` | L8 OLMoE 40w | 275KB |
| `results/prune_deconvention_L8_250win.json` | L8 OLMoE 250w | 1.5MB |
| `results/prune_deconvention_L8.json` | L8 OLMoE 250w (symlink to 250win) | 1.5MB |
| `results/prune_deconvention_L4.json` | L4 OLMoE 40w | 271KB |
| `results/prune_deconvention_L12.json` | L12 OLMoE 40w | 270KB |
| `results/prune_deconvention_L8_Qwen.json` | L8 Qwen 40w | 243KB |
| `docs/RESULTS_7_1_COMPLETE.md` | 本报告 | — |
