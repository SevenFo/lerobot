# LeRobot 训练配置参数指南

本文档详细介绍了LeRobot训练脚本 `lerobot/scripts/train.py` 中所有可配置的参数。训练脚本使用层次化的配置系统，通过 `TrainPipelineConfig` 类进行配置管理。

## 配置方式

### 1. 命令行参数
```bash
python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=lerobot/pusht \
    --env.type=pusht \
    --batch_size=16
```

### 2. 配置文件
```bash
# 从本地配置文件加载
python lerobot/scripts/train.py --config_path=path/to/config.json

# 从HuggingFace Hub加载
python lerobot/scripts/train.py --config_path=lerobot/diffusion_pusht
```

### 3. 配置文件 + 命令行覆盖
```bash
python lerobot/scripts/train.py \
    --config_path=path/to/config.json \
    --batch_size=32 \
    --steps=50000
```

## 核心配置参数

### 数据集配置 (`dataset`)

| 参数名               | 类型                    | 默认值   | 说明                                    |
| -------------------- | ----------------------- | -------- | --------------------------------------- |
| `repo_id`            | `str`                   | **必需** | 数据集的HuggingFace仓库ID               |
| `root`               | `str`                   | `None`   | 数据集存储的根目录                      |
| `episodes`           | `list[int]`             | `None`   | 指定使用的episode列表，None表示使用全部 |
| `image_transforms`   | `ImageTransformsConfig` | 默认配置 | 图像预处理变换配置                      |
| `revision`           | `str`                   | `None`   | 数据集版本/分支                         |
| `use_imagenet_stats` | `bool`                  | `True`   | 是否使用ImageNet统计数据进行归一化      |
| `video_backend`      | `str`                   | 自动检测 | 视频后端（如 'pyav', 'opencv'）         |

**示例:**
```bash
--dataset.repo_id=lerobot/pusht \
--dataset.episodes=[0,1,2,3,4] \
--dataset.use_imagenet_stats=false
```

### 策略配置 (`policy`)

| 参数名        | 类型   | 默认值   | 说明                                                             |
| ------------- | ------ | -------- | ---------------------------------------------------------------- |
| `type`        | `str`  | **必需** | 策略类型: `act`, `diffusion`, `tdmpc`, `vqbet`, `pi0`, `pi0fast` |
| `device`      | `str`  | `None`   | 设备类型: `cuda`, `cpu`, `mps`                                   |
| `use_amp`     | `bool` | `False`  | 是否使用自动混合精度训练                                         |
| `n_obs_steps` | `int`  | `1`      | 观察步数（历史帧数）                                             |
| `path`        | `str`  | `None`   | 预训练模型路径（用于微调）                                       |

**策略特定参数:**

#### ACT策略
- `n_action_steps`: 动作预测步数
- `chunk_size`: 动作块大小
- `hidden_dim`: 隐藏层维度
- `n_encoder_layers`: 编码器层数
- `n_decoder_layers`: 解码器层数

#### Diffusion策略
- `n_action_steps`: 动作预测步数
- `num_inference_steps`: 推理步数
- `down_dims`: 下采样维度列表
- `kernel_size`: 卷积核大小
- `n_groups`: 组归一化的组数

#### TDMPC策略
- `use_mpc`: 是否使用模型预测控制
- `horizon`: 预测时域
- `mppi_iterations`: MPPI迭代次数

**示例:**
```bash
--policy.type=act \
--policy.device=cuda \
--policy.use_amp=true \
--policy.n_action_steps=10
```

### 环境配置 (`env`)

| 参数名 | 类型  | 默认值   | 说明                                   |
| ------ | ----- | -------- | -------------------------------------- |
| `type` | `str` | `None`   | 环境类型: `pusht`, `aloha`, `xarm`, 等 |
| `task` | `str` | 环境默认 | 具体任务名称                           |
| `fps`  | `int` | 环境默认 | 环境帧率                               |

**常见环境:**
- `pusht`: PushT推拉任务
- `aloha`: ALOHA双臂机器人
- `xarm`: XArm机械臂
- `koch`: Koch机器人

**示例:**
```bash
--env.type=aloha \
--env.task=AlohaInsertion-v0 \
--env.fps=50
```

### 训练超参数

| 参数名        | 类型  | 默认值   | 说明               |
| ------------- | ----- | -------- | ------------------ |
| `batch_size`  | `int` | `8`      | 训练批大小         |
| `steps`       | `int` | `100000` | 总训练步数         |
| `num_workers` | `int` | `4`      | 数据加载工作进程数 |
| `seed`        | `int` | `1000`   | 随机种子           |

**示例:**
```bash
--batch_size=32 \
--steps=200000 \
--num_workers=8 \
--seed=42
```

### 优化器和调度器配置

| 参数名                       | 类型                | 默认值 | 说明                                |
| ---------------------------- | ------------------- | ------ | ----------------------------------- |
| `use_policy_training_preset` | `bool`              | `True` | 是否使用策略预设的训练配置          |
| `optimizer`                  | `OptimizerConfig`   | `None` | 自定义优化器配置（当预设为False时） |
| `scheduler`                  | `LRSchedulerConfig` | `None` | 自定义学习率调度器配置              |

**注意:** 当 `use_policy_training_preset=True` 时，会自动使用策略预设的优化器和调度器配置。

### 输出和检查点配置

| 参数名            | 类型   | 默认值   | 说明                   |
| ----------------- | ------ | -------- | ---------------------- |
| `output_dir`      | `Path` | 自动生成 | 输出目录路径           |
| `job_name`        | `str`  | 自动生成 | 任务名称               |
| `save_checkpoint` | `bool` | `True`   | 是否保存检查点         |
| `save_freq`       | `int`  | `20000`  | 检查点保存频率（步数） |
| `resume`          | `bool` | `False`  | 是否恢复之前的训练     |

**示例:**
```bash
--output_dir=outputs/my_experiment \
--job_name=act_pusht_v1 \
--save_freq=10000 \
--resume=true
```

### 日志配置

| 参数名     | 类型  | 默认值 | 说明                 |
| ---------- | ----- | ------ | -------------------- |
| `log_freq` | `int` | `200`  | 日志记录频率（步数） |

### 评估配置 (`eval`)

| 参数名           | 类型   | 默认值  | 说明                          |
| ---------------- | ------ | ------- | ----------------------------- |
| `eval_freq`      | `int`  | `20000` | 评估频率（步数），0表示不评估 |
| `n_episodes`     | `int`  | `50`    | 评估时运行的episode数量       |
| `batch_size`     | `int`  | `50`    | 评估时的并行环境数量          |
| `use_async_envs` | `bool` | `False` | 是否使用异步环境（多进程）    |

**示例:**
```bash
--eval_freq=5000 \
--eval.n_episodes=20 \
--eval.batch_size=10 \
--eval.use_async_envs=true
```

### WandB配置 (`wandb`)

| 参数名             | 类型   | 默认值      | 说明                                  |
| ------------------ | ------ | ----------- | ------------------------------------- |
| `enable`           | `bool` | `False`     | 是否启用WandB日志记录                 |
| `project`          | `str`  | `"lerobot"` | WandB项目名称                         |
| `entity`           | `str`  | `None`      | WandB实体/团队名称                    |
| `notes`            | `str`  | `None`      | 运行备注                              |
| `run_id`           | `str`  | `None`      | 特定的运行ID                          |
| `mode`             | `str`  | `None`      | 模式: `online`, `offline`, `disabled` |
| `disable_artifact` | `bool` | `False`     | 是否禁用artifact保存                  |

**示例:**
```bash
--wandb.enable=true \
--wandb.project=my_robot_project \
--wandb.entity=my_team \
--wandb.notes="Experimenting with ACT policy"
```

## 完整示例

### 1. 从零开始训练
```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=lerobot/pusht \
    --policy.type=diffusion \
    --env.type=pusht \
    --batch_size=64 \
    --steps=100000 \
    --eval_freq=10000 \
    --wandb.enable=true \
    --wandb.project=pusht_experiments \
    --output_dir=outputs/diffusion_pusht
```

### 2. 微调预训练模型
```bash
python lerobot/scripts/train.py \
    --policy.path=lerobot/act_aloha_sim_transfer_cube_human \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --env.type=aloha \
    --env.task=AlohaInsertion-v0 \
    --steps=50000 \
    --output_dir=outputs/act_finetuning
```

### 3. 恢复训练
```bash
python lerobot/scripts/train.py \
    --config_path=outputs/my_training/checkpoints/last/pretrained_model/ \
    --resume=true \
    --steps=200000
```

### 4. 使用多个数据集（当前不支持，但配置已预留）
```bash
# 注意：目前还不支持多数据集训练
--dataset.repo_id=[lerobot/dataset1,lerobot/dataset2]
```

## 配置文件格式

训练完成后，配置会保存为 `train_config.json` 文件，格式如下：

```json
{
    "dataset": {
        "repo_id": "lerobot/pusht",
        "episodes": null,
        "video_backend": "pyav"
    },
    "env": {
        "type": "pusht",
        "fps": 10
    },
    "policy": {
        "type": "diffusion",
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "num_inference_steps": 10
    },
    "batch_size": 64,
    "steps": 100000,
    "eval_freq": 10000,
    "wandb": {
        "enable": true,
        "project": "lerobot"
    }
}
```

## 性能调优建议

### 1. 数据加载优化
- 增加 `num_workers` 如果数据加载成为瓶颈
- 调整 `batch_size` 以充分利用GPU内存
- 使用 `pin_memory=True`（脚本自动处理）

### 2. 内存优化
- 启用 `policy.use_amp=true` 使用混合精度训练
- 根据GPU内存调整批大小

### 3. 训练监控
- 设置合适的 `log_freq` 和 `eval_freq`
- 启用WandB进行实验跟踪
- 定期保存检查点（`save_freq`）

### 4. 环境优化
- 评估时使用 `eval.use_async_envs=true` 可以加速
- 调整 `eval.batch_size` 匹配可用CPU核心数

## 常见问题

1. **如何选择合适的批大小？**
   - 从小批大小开始（如8），逐渐增加直到GPU内存用尽
   - 使用混合精度训练可以支持更大的批大小

2. **评估频率应该设置多少？**
   - 对于快速迭代：5000-10000步
   - 对于最终训练：20000步（默认值）
   - 设置为0可以禁用训练期间的评估

3. **如何恢复中断的训练？**
   - 使用 `--resume=true` 和指向检查点目录的 `--config_path`

4. **可以在训练时修改参数吗？**
   - 大部分超参数可以在恢复训练时修改
   - 策略架构参数通常不能修改（需要重新训练）

## 更多资源

- [策略配置详情](../lerobot/common/policies/)
- [环境配置详情](../lerobot/common/envs/configs.py)
- [数据集相关文档](../lerobot/common/datasets/)
- [WandB集成说明](../lerobot/common/utils/wandb_utils.py)
