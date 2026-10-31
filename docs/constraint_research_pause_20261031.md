# 约束后训练研究暂停记录

2026-10-31，用户要求暂停实验和开发，保存现有代码后重新打磨研究设计。本记录描述暂停时的事实，不发布新的数据集、不激活候选模板，也不构成继续训练或实验的授权。

## 研究意图

参考轨迹用于建立动作先验，后续优化目标逐步转向任务成功率。研究希望用低成本、训练时不运行仿真器、主要从 action 计算的成功代理，降低 RL 后训练的交互成本。约束是这一代理的首个实现；RSI/对抗反馈用于发现并改善代理的漏洞。

当前实现只是可组合的几何区间约束，并未训练一个能够从任意 action 判断任务成功的模型，也未训练 RL 约束 agent。大部分范围仍由参考特征 min/max 加容差产生，不能当作真实物体可行集合。

## 已有训练与开发成果

- 原训练集为 RoboTwin 原生专家产生的 50 个 clean 任务、每任务 50 条成功示范。RAX/程序化标注将约束嵌入 HDF5。
- 已完成的四卡、5 epoch、全局 batch 256 训练使用旧 schema 1.2 / `full_motion_v1`，共 132,759 条标注，其中 103,084 条为普通运动补全。该条目比例不是 loss 贡献比例。
- 当前主开发代码支持 schema 1.5、多步方向窗口、固定 EEF 局部点、参考阶段和双夹爪组合。代码能力与旧训练标签版本不能混同。
- 双瓶 V3 是未激活的开发候选，没有写回旧 HDF5，也没有用于新训练。已有 120 条开发回放的 CPU 重评分保留 81/81 成功、拒绝 36/39 失败，仍有 3 条已知严格反例。
- 后续三个新场景共 3 次 expert、6 次基线和 54 条构造挑战。54 条中 38 成功、16 失败，13 条为命令和实际轨迹均合规但任务失败。该结果是代理验证，不是模型策略成功率。
- 双鞋仅完成两条同场景 command-index 基线：native-ever 成功 2/2，最终状态成功 1/2。缺失的专家延时采样不能从 EEF 序列补造。
- 柜子观察报告 `IK_FAIL`。独立 12 次 solver-only 诊断发现原目标纯运动学有数值解，而 self-enabled 的返回解未同时满足目标与 aggregate feasibility。它不是物理不可达证明，生产碰撞设置未改。

## 已提交代码的分界

主分支 `feat/robotwin-constraint-supervision` 保存录制、数据内嵌、模板、离线评分、有限验证和参考阶段实现。OpenWAM 的对应分支为 `feat/constraint-supervision`。

以下分支保留独立变更，不合并或自动启用：

| 分支 | 内容 |
| --- | --- |
| `feat/angle-center-hint` | 角度中心的固定参考梯度提示；与 OpenWAM 同名分支配套 |
| `fix/handover-mic-duplicate-height` | 重复高度约束修订 |
| `feat/fixed-reference-native-branches` | 固定参考事件确定的原生分支 |
| `feat/expert-terminal-evidence` | 专家终局只读证据 |
| `fix/control-timing-text-encoding` / `feat/control-timing-stagea` | 时序诊断与编码兼容 |
| `feat/scale-fresh-success-metadata` | 源绑定的初态元数据 |
| `codex/expert-failure-observation-v1` | 有限专家失败观察 |
| `codex/planner-status-codec-v2` | 有界状态数组保存与重验证 |
| `codex/planner-result-observer-v1` | 只读已有 CPU 求解器返回信息 |
| `codex/cabinet-targeted-v3` | 一次性柜子针对观察集成 |

`archive/constraint-research-20261031` 另存工作区外的独立实验源码。它是源代码快照，不是可直接部署的完整数据包。

## 门禁与统计口径的更正

后加的 29 场景、95% 置信度、10% 风险上界与 crafted/model 双来源协议，属于未来独立验证设计，不是 meta loss 或数据格式的固有要求，也不是 RoboTwin 官方要求。实际旧冻结自动流程没有启用该模块；旧新数据/训练流程因原 pilot 未过及 publication hold 而停止。

50 个任务已有源码审查；常规开发动作回放覆盖 48 个，另双鞋有受限诊断，因此原始动作回放涉及 49 个任务。这些覆盖数不是任务验收通过数。正式独立 holdout 尚未启动。

## 暂停状态

已停止本地周期巡检；没有运行中的本项目 GPU、仿真或 solver 作业。未修改原训练数据、权重、账号或 HF 配置。新姿态包只保留 CPU 准备文件，未部署或执行。

原始实验输入、逐条结果、冻结包及完整证据保留在原工作区和服务器，未随源码归档发布。后续研究方向与是否恢复实验，由用户重新确定。
