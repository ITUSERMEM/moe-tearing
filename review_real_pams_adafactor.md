# 审查报告: 真模型全参数 Adafactor PAMS 微调 (外部效度验证)

日期: 2026-07-11
脚本: expansion_32gb_128gb/scripts/real_pams_adafactor.py
模型: OLMoE-1B-7B-0125 (非论文 canonical 0924)
证据等级: E (相关性实验) + I (机制推断)
第 16 个训练 forward, 用户授权 (走 Adafactor 全参数方案)

## 1. 变更文件

- 新建 `expansion_32gb_128gb/scripts/real_pams_adafactor.py`: Adafactor 全参数 7B PAMS 微调
  - gradient_checkpointing_enable() 减激活 (全参数训练必须)
  - 全参数解冻 (6.9B), torch.optim.Adafactor (factored 内存高效, state~1.5GB)
  - PAMS 重算绕过 hook/checkpoint 冲突: forward 后用 captured mlp 输入 h (detach fp32) + gate.weight 重算 gate logits 算 pams_reg, 梯度独立于 checkpoint 经 gate.weight 回传
  - measure_full: A 口径 fp32 基线=当前模型转 fp32 覆盖所有权重 (全参数口径, 非仅 gate)
  - 诊断: NaN/Inf 检测 + grad_norm 监控
- 新建结果 JSON:
  - results/real_pams_adafactor_selftest5.json (lr=1e-5 eps=1e-30, step4 NaN)
  - results/real_pams_adafactor_diag20.json (lr=1e-5 eps=1e-3, 20步权重没动)
  - results/real_pams_adafactor_diag_lr1e2.json (lr=1e-2, 20步 d_t 微降)
  - results/real_pams_adafactor_lr1e2_300.json (lr=1e-2, 300步正式)

## 2. 实验结果 (300步 lr=1e-2 λ=0.05 ε=0.0116 clip=1.0)

| 指标 | before | after | 判读 |
|------|--------|-------|------|
| L8 dt_med | 0.0742 | 0.0654 | **降11.8% (非推大!)** dt_push=-0.118 |
| L8 flip_A | 5.98% | 5.59% | 降6.5% (不显著) |
| L12 flip_A | 5.90% | 4.37% | 降25.9% |
| L4 flip_A | 4.18% | 5.90% | **升41% (跨层危害)** |
| held-out loss | 2.7827 | 2.8478 | **增2.3%** |
| L8 frac<ε | 0.1082 | 0.0977 | 左尾推大 (中位降) |
| pams_reg | 1.5548 | 1.3607 | 降12% (只推左尾) |

verdict = FULL_PARAM_STILL_NOT_WORKING

## 3. 机制结论 (彻底坐实)

1. **d_t 中位降而非推大**: 全参数300步 PAMS 没推大 margin 中位反而降11.8%, 与 probe 的 10.95× 膨胀完全相反。PAMS 稀疏正则只作用 d_t<ε 左尾~11%, 中位 token (d_t~0.075) 不在作用集被 CE 梯度主导拉低。
2. **全参数比仅 gate 更糟**: 仅 gate d_t 中位"不变", 全参数 d_t 中位"降11.8%"——CE 对全参数拉低 margin 更有效, PAMS 左尾正则更弱无法对抗。frac<ε 降说明 PAMS 确推大左尾, 但中位被 CE 压低, 净效果中位降。
3. **跨层危害复现**: L4 flip 升41% (全参数改动跨层传播, 与 gate+L8 expert 的 L12 升7.6× 同机制)。
4. **三设置全失败**: 仅gate升3× / gate+L8expert L12升7.6× / 全参数L4升41%+中位降11.8%+loss增2.3%。probe 全参数成功 (10.95×) 是小模型+全参数+PAMS主导; 真模型大+CE主导+PAMS弱, 即使全参数也无法全局推大 margin。

PAMS 真模型外部效度未复现 (E), 机制问题彻底坐实。PAMS 是 probe-level 方法, 真模型受规模+精度限制不适用。

## 4. 工程突破 (可复用)

- **Adafactor eps=(1e-3,1e-3) 解 bf16 全参数 NaN**: eps[0]=1e-3 是 1/sqrt(rms) 分母, 防 rms 极小时步长放大爆。lr=1e-4 eps=1e-30 step1 NaN, lr=1e-5 eps=1e-30 step4 NaN, eps=1e-3 后 300步稳。
- **显存28.5GB可行**: Adafactor factored state ~1.5GB (非预估25GB), 32GB 卡够全参数7B训练+checkpoint。
- **PAMS 重算机制工作**: forward后 captured h detach + gate.weight 重算 logits, 梯度经 w.float() cast 回传 bf16 gate.weight, 独立于 checkpoint。dummy 验证 w.grad bf16 非零。
- **bf16 全参数训练 update 需过阈值**: grad_norm 40 + clip 0.5 缩80× → update ~6e-8 被 bf16 吞 (权重不动); clip 1.0 + lr 1e-2 → update ~1.25e-4 刚过 bf16 阈值, 权重动。

## 5. 口径声明

- OLMoE-1B-7B-0125 非 0924 canonical, 不可混。
- 全参数 7B + Adafactor (非 AdamW) + gradient checkpointing + PAMS 重算(非hook)。
- A 口径 fp32 基线=当前模型转 fp32 覆盖所有权重 (全参数口径)。
- 全参数 SFT 级 300 步非 fully-converged。

## 6. 反数据修补

失败结果如实降级报告, 不重写为支持 PAMS。三设置全失败的负面证据充分, 写入论文 limitation。

## 7. 后续方向

1. 论文 limitation 写入: PAMS probe-level 方法, 真模型外部效度受规模+精度+CE主导限制, 三设置(仅gate/gate+expert/全参数)全失败。
2. PAMS 定位为 future work (小模型方法论验证), 不作为真模型主张。
3. 不再追加真模型 PAMS 实验 (三设置已穷尽: 仅router/局部expert/全参数)。

链接: [[real-pams-finetune-findings]], [[real-pams-eps-calib-findings]]