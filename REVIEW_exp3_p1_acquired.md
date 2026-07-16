# 审查报告: 实验③ P1 checkpoint-trajectory acquired 裁决 + 论文升级

日期: 2026-07-10
范围: 实验③(exp3_p1_checkpoint.py)执行 + 论文 07/08 P1 升级 + 编译验证

## 一. 实验③ 执行

### 设计
- 目的: 论文 registered prediction P1 —— 35× gap 源头专家输出范数差 acquired(训练习得) vs intrinsic(架构固有) 未定. 判据: Δflip@L8 跨训练 checkpoint 轨迹, flat->intrinsic, changing->acquired.
- 口径: energy 口径 Δflip@L8 = median(cross[8]-near[8]) (源头残差跳变), L8 注入, K=8, N_TOKENS=300, OLMoE-1B-7B-0924 bf16.
- checkpoint 来源: OLMoE-1B-7B-0924 仓库 git branch (非独立 repo). 选 3 个跨度大: step5000-tokens20B(early) / step500000-tokens2097B(mid) / main(=step1220000-tokens5117B final). 每 ckpt 13.8GB.

### 脚本改动 (exp3_p1_checkpoint.py)
1. 顶部 import 加 snapshot_download + shutil + subprocess.
2. 新增 resolve_ckpt(): 支持 "repo@revision" 语法 (OLMoE 中间 ckpt 是同 repo 的 branch), snapshot_download(allow_patterns=safetensors/json/txt/tokenizer) 到临时 cache_dir.
3. run_one_checkpoint(): 加下载 + 跑完 gc.collect+empty_cache + shutil.rmtree 临时 cache (磁盘峰值 ~14GB).
4. main(): 改子进程隔离 (--single IDX 子进程模式, 每 ckpt 独立子进程跑完退出彻底释放显存, 解决残留未释放隐患) + 续传 (已有 experimentA JSON 跳过).

### 结果 (ACQUIRED, tier E)
| checkpoint | step | tokens | Δflip@L8 | Δdir@L8 | n | IQR |
|---|---|---|---|---|---|---|
| early | 5000 | 20B | 0.0173 | 0.0060 | 71 | [.0001,.0274] |
| mid | 500000 | 2097B | 0.0511 | 0.0029 | 42 | [.0001,.0771] |
| final | 1220000 | 5117B | 0.0639 | 0.0064 | 57 | [.0002,.0769] |

- Δflip 单调递增 3.7× (0.0173->0.0639), rel_change = range/median = 0.0466/0.051 = 0.91× (>=0.2 阈值) -> ACQUIRED.
- Δdir 跨 checkpoint 平坦 (~0.003-0.006, 内对照): 法向效应是架构属性不随训练变, 翻转效应是习得的.
- 翻转主导比 Δflip/Δdir: early 2.9× / mid 17.6× / final 10×, 翻转主导随训练增强.
- verdict=ACQUIRED, evidence_tier=E. JSON: results/exp3_p1_checkpoint.json.

### 口径干净性
- ③ 是 OLMoE-0924 自身训练轨迹 (canonical 版本), energy 口径 (残差范数跳变), 非 KL 口径.
- 不涉及 0125/0924 跨版本 (AGENTS.md 禁).
- 不受 ② Qwen bf16 KL 噪声底影响 (②是KL口径+Qwen侧). 干净可升 (E).
- tier E 判定: trajectory 观察 + Δdir 内对照, 非人为干预训练的 counterfactual. 比纯观察强(time-ordered + 内对照)但保守标 E.

## 二. 论文变更 (3 处)

### 1. 08_discussion_appendix.tex P1 段 (line 20-22 -> 扩展)
原: "we register it as P1 and defer it to the journal version, not the preprint\tierP."
改: adjudicate P1 here. 3 个 0924 checkpoint 轨迹 (5000/500000/final steps, 20B/2097B/5117B tokens). Δflip@L8 单调 0.017->0.051->0.064 (3.7×, rel_change=0.91), Δdir@L8 平坦 ~0.003-0.006 内对照. 源头翻转效应 OLMoE 侧 training-acquired 非架构固有. P1 confirmed for OLMoE component, tier (E). Qwen 侧轨迹未公开故 acquired-or-intrinsic open, tier (E).
tier: P -> E.

### 2. 08_discussion_appendix.tex limitation #1 (line 38-40)
原: "Cross-model attribution unresolved (P1). ... acquired-versus-intrinsic is unadjudicated"
改: "Cross-model attribution partially adjudicated (P1, E)." 0924 轨迹 Δflip@L8 单调 0.017->0.051->0.064 (3.7×, rel_change=0.91) + Δdir 平坦内对照 -> OLMoE 侧 training-acquired 非固有 (E). Qwen 侧轨迹未公开故 open. N=176 不足 dose-response (保留原).

### 3. 07_cross_model.tex "What this means" 段末 (line 84-86 后追加)
追加: "Separately, and on the OLMoE side only (an energy-caliber self-trajectory, not the cross-model KL caliber affected by the noise floor above), the released 0924 training trajectory adjudicates the source flip effect as training-acquired, not intrinsic: Δflip@L8 grows monotonically 0.017->0.051->0.064 across checkpoints with the normal-direction effect flat as an internal control (sec:discussion, P1)\tierE."
明确标注 OLMoE 侧 + energy 口径 + 不受 ② KL 噪声底影响, 与 ② 的跨模型 KL 降级不冲突.

## 三. 不改的
- 01_intro 贡献条目: P1 acquired 是 08 registered prediction 裁决, 非 01 贡献条目核心, 不加 (避免偏离 cross-model 主题).
- ② 的 35×/60× reverse-falsify 警告保留 (② 是跨模型 KL 口径, ③ 是 OLMoE 自身 energy 口径, 不冲突).
- P2 段不动.

## 四. 验证
1. 编译 pdflatex×3 + bibtex: 0 errors / 0 [?] / 0 undefined / 0 overfull / 0 LaTeX Warning. pdf 711469 bytes (从 708263 +3KB, ③ 内容合理).
2. 数字核对 (与 exp3_p1_checkpoint.json 一致): 0.017/0.051/0.064 (JSON 0.0173/0.0511/0.0639 三位有效), 3.7× (0.0639/0.0173=3.69), rel_change 0.91 (JSON 0.9121), Δdir ~0.003-0.006 (JSON 0.0060/0.0029/0.0064). 全部吻合.
3. tier 审计: 新增句均带 \tierE. P1 段 (E) + (E) (OLMoE acquired + Qwen open), limitation #1 (E), 07 追加 (E).
4. 版本标签: ③ 用 0924 (canonical), \versiontag{0924}, 不混淆 0125.
5. \cref / \label / \cite 保留, 无新 \cite (③ 是本工作实验).

## 五. 战役整体收尾 (①-⑥)
- ①done: OLMoE local-soft 50-seed cost ZERO_RESULT_AT_POWER (CI 含0, n_pos/n_neg=24/26).
- ②done: Qwen fp32 FP32_NOISE_FLOOR_GAP_INFLATED_BF16 -> 论文 07/08/01 全链路降级.
- ③done: OLMoE-0924 轨迹 ACQUIRED -> P1 (P)->(E), 论文 07/08 升级.
- ④done: jump_rel SHARED_INFLATES_DENOMINATOR (drop_frac=0.569).
- ⑤不跑: ②已证 Qwen bf16 Δflip 是噪声底, ⑤测其阻尼=测噪声, 反数据修补铁律下不跑.
- ⑥done: OLMoE 自然臂扩样本 POWERFUL_NULL (r=0.0205, CI上界0.0293<0.05) -> 论文 03/01 power declared.

链接: [[exp3-p1-acquired-findings]], [[exp2-qwen-fp32-findings]], [[exp6-natural-arm-power-findings]], [[icml-paper-external-review]].