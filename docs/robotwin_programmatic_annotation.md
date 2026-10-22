# RoboTwin 程序化约束标注

原流程对每条 HDF5 独立启动 Codex（`xhigh`），重新读图、分析阶段、写 Python 计算数值、生成约束和报告。前四条端到端约 4–6 分钟。GPU 没有参与这些重复推理。

现在采用任务模板与逐轨迹数值实例化：

1. 本机 RAX 审核一个尚未覆盖的任务/事件模式，输出该条轨迹的 `annotations.json`、`REPORT.md` 和 `task_template.json`。
2. 模板只有数据和规则，不执行任意 Python，不包含固定帧号、参考示范坐标或允许区间。模板允许选择七种现有几何原语，以及参考事件、夹爪稳定区间、终点和轴向位置极值。
3. H200 用至少三个不同 seed、不同初始几何的成功 clean 轨迹验证模板，再将其状态从 candidate 改为 active。
4. 程序逐条读取真实 EEF 位姿和夹爪指令，重新计算左右臂、事件帧、参考坐标、区间和数值范围；继续执行原有来源哈希、schema、事件及参考值校验，原子写入 HDF5 后发布。
5. 未见过的事件模式、缺少必要锚点或验证失败交还 RAX。正在由 agent 处理的轨迹不会被程序抢占。每任务同一时间最多一个 agent，避免重复开发模板。

这里的 active 仅表示规则在不同参考轨迹上可执行且标注有效，并不证明它表达了所有任务成功条件。前三个预置模板从已有 RAX 报告提取可复用的特征；无法稳健选择的手工阶段暂不迁移。例如，瓶子模板保留闭合位置、闭合指令、终点高度及闭合到终点的竖直位移，不自动继承某一示范特有的晚期姿态区间。

## 范围和训练语义

所有边界使用既有显式示范包络先验：位置 5 mm、角度 5°、夹爪指令比例 0.02。它们不是通过该程序获得的物理可行范围。模板不能自行扩大这些常量。已做过的仿真容差校准仍需按任务、参考系和后续轨迹条件独立适用。

当前动作适配器读取 absolute / world / WXYZ EEF 与绝对夹爪指令比例。训练 loss 仍使用原有七种可微几何约束和 reference-only 激活，不改变 OpenWAM schema。位移保留原始起点锚点，跨 chunk 缺少起点时必须由训练端屏蔽，不能重新绑定到 chunk 起点。

模板的 `signatures` 对左右臂对称变体使用排序后的逐臂夹爪事件序列；不会按时间比例拉伸参考帧。新的事件序列需要新的审核。缺少参考事件、零长度位移、错误四元数/单位轴、不合法范围等均不发布。

## 已执行的验证

2026-10-22 首批三个模板在真实数据上通过：

| 任务 | 不同参考轨迹数 | 数值实例化中位耗时 |
| --- | ---: | ---: |
| adjust_bottle | 21 | 8.3 ms |
| beat_block_hammer | 30 | 6.8 ms |
| click_alarmclock | 31 | 6.7 ms |

首批 80 条程序化发布包含来源哈希、校验、HDF5 复制/更新及队列写回，总处理耗时约 **4.38 秒**，单条中位 **52.9 ms**，最大 **87.3 ms**。这是已激活模板的处理时间，不包含首次 RAX 审核、等待新轨迹和仿真录制。后续任务可能更长，不能直接把这一数字当作整个数据集的工期。

回归覆盖帧数变化、坐标平移、左右臂切换、未知事件序列回退、单位/边界检查、三独立场景激活门槛、HDF5 发布以及 agent 租约互斥。

新增 RAX 任务审核链路也已实跑：`grab_roller` 由本机 worker 自动生成模板，校验后上传 H200，通过三场景门槛并激活；首条包含 8 个双臂几何约束，其余匹配轨迹自动交给数值执行器。该任务不再需要每条重新调用模型。相关单元测试共 **44 项通过**。

随后 `handover_block` 也自动完成首次审核，生成 10 项约束并激活模板。后续快照共有 5 个活跃任务模板、121 条程序化发布，发布中位耗时 57.6 ms。此时数据集成功录制 218 条、已标注发布 131 条，已发布 HDF5 的约束和独立初始场景审计通过。

## 运行与审计

数据根目录仍是 `/mnt/data/liangjiaqi/rax_train/datasets/robotwin_clean50_v1`。

- `annotation_templates/<task>/<sha256>.json`：规则、来源、三场景验证记录和 candidate/active/rejected 状态。
- `programmatic_publications.jsonl`：每条发布路径、模板哈希和耗时。
- `programmatic_status.json` / `programmatic.log`：常驻程序进度和日志。
- `annotation_control.json`：`task_templates: true` 启用混合领取；`pause_agent_claims` 暂缓新 agent 请求，`paused_workers` 可单独切换 worker。
- HDF5 的约束 evidence 含模板哈希；修订历史的 `agent_run` 使用 `template:<sha256>`。RAX 首次报告和模板一同保存在远端 review 目录。

程序化服务只需要 CPU，不加载模型：

```bash
/mnt/data/liangjiaqi/yzm/miniconda3/envs/RoboTwin/bin/python \
  scripts/run_robotwin_programmatic_annotation.py \
  --root /mnt/data/liangjiaqi/rax_train/datasets/robotwin_clean50_v1
```

本机两个 `run_robotwin_annotation_worker.py` 继续使用本机 Codex 登录。新的 worker 默认要求输出任务模板；凭据不发送到 H200。agent 模板格式在 `agent_skills/control/robotwin/geometric-constraints/references/programmatic-template.md`。

本次检查同时定位了原生专家在 `click_alarmclock` seed 34、`grab_roller` seed 21 的空抓取位姿传播错误。收集器保留原生候选/回退逻辑，仅将已观察到的空位姿索引或 move 断言转成该 seed 的可重试失败，不吞掉其他编程错误。调度器重启时接管存活的 collector，保留现有轨迹，并重新排队未达 50 条的任务。

两个原始失败 seed 已在 H200 单独重跑，均实测转换为 `native_grasp_pose_unavailable` 可重试失败；没有修改 RoboTwin 原生专家、接受失败轨迹或占用成功轨迹名额。
