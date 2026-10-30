# 原生回放搜索约束反例

入口是 `scripts/search_robotwin_constraint_counterexamples.py`。它搜索“动作满足
固定约束、控制器确实执行、RoboTwin 原生任务判定仍失败”的轨迹，输出开发证据，
不会自动批准约束、发布数据或开始训练。离线筛选只需要 CPU；原生回放需要
RoboTwin 环境及 GPU。

## 固定输入与启动

每个 campaign 使用一个已封存的专家参考场景。参考负责初始状态、约束参数及
激活帧，不限制候选必须沿专家轨迹运动。可输入构造轨迹、真实模型轨迹，以及
上一轮已核验的反例。帧 0 保留实测状态；所有候选固定 N 帧，不做重采样、补齐
或 DTW。任务判定采用同一固定 N−1 条命令内的原生 `ever_success`，不是标准
benchmark 最大步数内的成功率。

```bash
python scripts/search_robotwin_constraint_counterexamples.py \
  --reference /evidence/references/seed10000.json \
  --reference-manifest /evidence/experiment_manifest.json \
  --template /revisions/lift_pot.json \
  --training-root /datasets/robotwin_clean50_v1 \
  --split dev --split-plan /plans/lift_pot.json \
  --candidate-budget 24 --baseline-repeats 2 \
  --model-candidates /model_traces/lift_pot/model_candidates.json \
  --output /experiments/lift_pot_dev_v1
```

`--check-only` 只封存与校验输入，不启动 GPU。它仍会冻结模型文件 hash；正式
模型文件生成前，请使用独立临时目录检查，不要冻结正式输出再替换输入。
重新执行已完成目录只校验证据。失败或未提交的原生 trial 不能在原目录偷偷重跑；
调查原因后新建 campaign，保留原结果。

split plan 的 seed 必须读取参考的 `provenance.seed`，不能由文件名猜测。例如
本轮旧证据的 `adjust_bottle`、`lift_pot` 首个 seed 为 10000，
`beat_block_hammer` 为 10001。下例只适用于 `lift_pot`：

```json
{
  "version": 1,
  "task": "lift_pot",
  "dev_seeds": [10000],
  "holdout_seeds": [11000, 11001, 11002]
}
```

开发、最终留出和训练 seed 必须互斥。参考的原文件及原 manifest 原样留存；
`reference.json` 是当前显式模板重新实例化的工作标签。它单独记录原标注代码
hash，绝不会把旧参考伪装成当前代码生成的记录。

## 候选、模型及反馈

候选生成器为 `services/robotwin/adversarial_search.py`，包含停顿后跳跃、捷径、
阶段重排、双臂不同步、提前闭合、夹爪抖动、工具轴旋转、抓取位置偏移及其后续
补偿等组合。候选预算为去重后的数量上限；生成器在固定随机种子下先保持来源和
结构多样性，再优先筛选满足约束的轨迹。原生结果不能反向挑选场景 seed。

模型导入文件结构如下，`trajectory` 使用双臂世界坐标 XYZ + WXYZ 和 [0,1]
夹爪开度，完整 `action_contract` 从参考 bundle 复制：

```json
{
  "version": 1,
  "task": "lift_pot",
  "scene_seed": 10000,
  "scene_sha256": "<参考场景 SHA256>",
  "alignment": "reference_command_index_v1",
  "action_contract": {},
  "candidates": [{
    "candidate_id": "openwam_closed_loop_0",
    "source": "model",
    "family": "model_closed_loop",
    "trajectory": {
      "left": {"pose": [], "gripper": []},
      "right": {"pose": [], "gripper": []}
    },
    "model_provenance": {
      "source_kind": "model_closed_loop",
      "checkpoint_sha256": "<真实 checkpoint SHA256>",
      "source_run_sha256": "<真实模型运行 SHA256>",
      "full_policy_rollout": true
    }
  }]
}
```

上面的空数组和空 contract 是结构占位，不能直接执行。闭环来源要求每一步来自
真实模型及当前观测。`model_receding_horizon_snapshot` 只表示固定观测锚点的一次
预测，需要声明 `full_policy_rollout=false`、原 chunk hash、`embedding` 中的
`chunk_start/chunk_end/prefix_source/suffix_source`，不能作为完整策略成功率证据。

生成器可能把模型候选与参考混合来寻找允许集合边缘。它保留原预测 hash、
`search_transform` 和混合比例；派生动作的 `candidate_is_full_policy_rollout`
为 false。原生边界夹爪裁剪等变换也必须保留在模型 provenance。本工具的 pass
仅指实际发给原生环境的有效命令，`raw_model_constraint_pass` 为 null，不声称
原始未裁剪模型输出也合规。

下一轮可通过 `--feedback feedback.json` 引入已完成开发 campaign 的强反例：

```json
{
  "version": 1,
  "campaigns": [{
    "campaign_root": "/experiments/lift_pot_dev_v1",
    "trial_ids": ["candidate_0000"]
  }]
}
```

反馈会重新核验原证据，要求相同场景、seed 和参考轨迹。新模板已经拒绝的父轨迹
会留有 receipt，但不再作为“当前允许”的父候选。最终留出评估禁止反馈。

## 如何解读结果

每个场景先运行两个专家命令基线，确认原生执行与任务成功。开发发现阶段允许
基线实际轨迹被标签误拒绝，但会单独报告 `baseline_false_rejects`；基线控制失败
或任务失败会阻断搜索。私有 replay adapter 核验 URDF 和规划器固定基座 SE(3)，
不会修改 RoboTwin 全局源码。

| 字段 | 含义 |
|---|---|
| `commanded_constraint_pass` | 发送给控制器的轨迹满足冻结标签 |
| `actual_constraint_pass` | 实测执行轨迹满足同一标签及参考激活 |
| `eligible` | 全程完成、双臂各帧规划成功、绝对跟踪及相对干预均实现 |
| `is_counterexample` | eligible + 命令合规 + 原生任务失败 |
| `strong_counterexample` | 上述条件成立，且实测轨迹也合规 |
| `command_only_counterexample` | 命令合规且执行达标，但实测轨迹越过约束边界；需检查边界与控制误差 |
| `controller_failure` | 命令合规，但规划/执行/实现检查未通过；不作为任务失败反例 |

`counterexample_summary.json` 包含进度、分类计数和逐 trial 账本。
`verify_campaign(path)` 从封存输入、原生结果和轨迹重新计算，不信任缓存的
成功率或 loss。所有产物都绑定 hash；跨 campaign 聚合需用
`manifest_sha256 + ':' + trial_id` 标识，避免不同 scene 的 trial ID 重复。

`scene_trace.json` 每 4 条命令记录一次物体/任务关节位姿，并包含首末帧。
它只用于解释夹空、碰撞、滑落等原因，绝不进入 action-only 约束计算或训练。
执行失败不能因为诊断图看起来失败就重新标成有效任务负例。

## 开发与最终验收隔离

开发默认只原生回放约束允许的候选，以固定预算提高找洞效率。这是对抗搜索，
不是独立同分布成功率估计；没有找到反例也不意味着约束充分。

最终 `--split holdout` 还需要 `--holdout-release`：其中 task、模板 hash、标注
policy hash、split plan hash、开发报告 hash 与封存时间必须匹配。
`split_plan.holdout_generator` 冻结生成器代码 hash、随机种子、候选预算、两个
基线及每来源预算；`include_outside_controls` 必须为 true，所有候选都进入原生
回放。这样可以同时衡量合规动作失败与成功动作误拒绝，不能只回放筛选通过者。
holdout 候选只能在冻结后生成，禁止使用开发反馈或调参后重复使用同一留出集。

最终是否允许发布由 `services/robotwin/constraint_refinement.py` 的独立门禁决定。
统计单位是独立场景及冻结来源分布，同一场景几十条扰动不能当作几十个独立样本。
本入口始终输出 `acceptance_ready=false`。
