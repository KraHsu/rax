# RoboTwin clean-50 采集

本批次覆盖原生 50 个任务，每任务收集 **50 条成功轨迹**，共 2,500 条。使用 `demo_clean` / aloha-agilex；物体布局、朝向和原生任务允许的物体实例随 seed 变化，背景、光照等采用 clean 配置。

## 数据流与质量检查

1. 原生专家规划，成功后保存关节轨迹。
2. 用同一 seed 重建场景，检查初始场景指纹一致，使用原生回放、录制和 HDF5 合并函数。
3. 再次检查原生任务成功。失败、初始状态重复和回放失配均不占 50 条名额，改用新的 seed。
4. 成功轨迹先存入 `raw/`，附带动作格式、seed、初始场景位姿、场景指纹及待标注约束容器。
5. 本机两个 RAX/Codex worker 审核任务与新事件模式并产出模板。H200 对模板覆盖的轨迹独立计算几何约束；未覆盖情况交回 RAX。详见 [程序化标注](robotwin_programmatic_annotation.md)。
6. H200 校验标注来源哈希、约束 schema、参考事件和参考轨迹是否落在允许集合内。至少有一条 accepted 约束后才发布到 `data/`。约束及修订历史嵌入该 HDF5。

同一任务内同时检查 seed 和**真实初始几何**不重复。场景指纹不包含 seed 或时间戳；物体位置按 1 mm 量化，四元数及非机器人关节位置按 1e-3 量化。四元数正负号统一。机器人本体不参与任务关节去重。完整未量化初始场景保存在 HDF5 的 `initial_scene/json`。

RAX 的 accepted 标注表示通过格式与参考轨迹检查。未单独进行物理校准的容差仍是明确的示范包络先验，不能当成该任务已验证的物理成功边界。不同任务、不同后续轨迹条件不会自动继承 `adjust_bottle` 校准的范围。

## 路径与运行方式

H200 数据集根目录：

```
/mnt/data/liangjiaqi/rax_train/datasets/robotwin_clean50_v1/
  batch_manifest.json
  <task>/aloha-agilex_clean_50/
    collection_status.json
    clean_manifest.json
    scene_info.json
    raw/                    # 成功但待标注
    data/                   # 约束校验通过，可供训练
    annotation_queue/       # 每条轨迹的标注状态
    attempts/seed<N>/       # 初始状态、原生轨迹/视频、录制收据或失败原因
```

`completed_recordings` 是成功录制数；`data/` 文件数才是标注完成数。每任务两者分别封顶为 50。`clean_manifest.json` 中的路径是录制时的原始位置；发布后的路径与约束修订在对应 `annotation_queue/episode<N>.json` 中。

批量调度器在首个任务产出两条不同初始状态的成功轨迹后展开至 8 GPU；每个 GPU 同时负责一个任务。任务完成后自动取下一个任务。每个任务有排他锁，已有成功轨迹会在续录时重新检查并计数。

只复用相同配置、基座位姿和关节顺序的 cuRobo 模型/求解器对象。场景、机器人实体与场景关联的 MPLib planner 仍重新创建；每次规划使用原生 solver reset 和当前关节状态，不复用上一条轨迹的初始场景。

启动或续录（H200）：

```bash
export ROBOTWIN_PATH=/mnt/data/liangjiaqi/RoboTwin
/mnt/data/liangjiaqi/yzm/miniconda3/envs/RoboTwin/bin/python \
  scripts/run_robotwin_collection_batch.py \
  --output /mnt/data/liangjiaqi/rax_train/datasets/robotwin_clean50_v1 \
  --episodes 50 --gpus 0 1 2 3 4 5 6 7 --pilot-first
```

查看成功录制、已标注、唯一场景数及错误（H200）：

```bash
/mnt/data/liangjiaqi/yzm/miniconda3/envs/RoboTwin/bin/python \
  scripts/robotwin_collection_status.py \
  --root /mnt/data/liangjiaqi/rax_train/datasets/robotwin_clean50_v1 --audit
```

本机标注 worker 的日志、任务输入、模型报告和发布收据在
`runtime/clean50_annotation/w0/` 与 `w1/`。本机运行时使用已配置的 Codex 登录及代理；凭据不复制到 H200。单次标注超时 30 分钟，单条最多尝试 3 次，失败记录保留在队列中。

另有 H200 常驻 `run_robotwin_programmatic_annotation.py` 消费已激活模板，按轨迹重新计算参数并写入同一 HDF5。首次 RAX 审核耗时与纯程序化发布耗时分别统计。

## 启动时已验证

首个任务 `adjust_bottle` 的 seed 0、1 均完成原生成功规划和成功回放，分别有 133、144 帧。瓶子初始位置分别为：

```
seed 0: [-0.0969740, -0.1280021, 0.7753564] m
seed 1: [-0.1187832, -0.0961892, 0.7737829] m
```

对应初始场景指纹不同。首条轨迹规划/回放/合并约 85.7/65.6/3.9 秒，第二条约 8.4/52.9/17.0 秒；这些是启动阶段实测耗时，不是全任务工期估计。

2026-10-22 启动后的审核快照：8 个任务均已产出成功轨迹，总计 39 条；各任务已录制数与唯一 seed、唯一初始场景数一致。`adjust_bottle` 的前两条已由两个本机 RAX worker 完成标注并发布到 `data/`，首条包含 4 项 accepted 约束。该快照不代表整批采集完成；实时数量以状态脚本为准。

采集、标注、约束 schema 和容差相关的 33 项单元测试通过。
