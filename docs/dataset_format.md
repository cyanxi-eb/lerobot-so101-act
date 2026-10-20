# 数据集格式（LeRobot v3 子集）

本项目**没有**依赖 `lerobot` 包（2 核 CPU 上装官方包依赖过重），而是按
LeRobot v3 的目录约定与字段命名实现了自己的 writer / reader。本文档说明：

1. 实际写出的磁盘布局；
2. 与官方 LeRobot v3 的对应关系；
3. 子集差异（哪些字段有、哪些没有）；
4. 如何验证一个数据集。

## 1. 磁盘布局

```
data/demos_act/
├── meta/
│   ├── info.json              # 数据集元信息 + features schema + generation 说明
│   ├── episodes.jsonl         # 每条 episode 一行：长度 / 任务 / 全局帧区间
│   ├── episodes_meta.jsonl    # 子集扩展：每条 episode 的场景/域随机化参数
│   ├── tasks.jsonl            # task_index -> 语言任务字符串（UTF-8，中文）
│   └── stats.json             # 全量统计量（state / action / 图像逐通道）
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
└── videos/
    └── chunk-000/
        ├── observation.images.front/episode_000000.mp4
        └── observation.images.wrist/episode_000000.mp4
```

数据规模（本项目实测）：

| 数据集 | episodes | frames/ep | 总帧数 | 分辨率 | fps | 磁盘 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `data/demos_act` | 80 | 60 | 4800 | 96×96×3 ×2 相机 | 7.5 | ≈2.6 MB |
| `data/demos_lang` | 80 | 60 | 4800 | 96×96×3 ×2 相机 | 7.5 | ≈2.9 MB |

`fps = 7.5` 来自「60 个控制步 = 8.0 s 单次抓取循环」，见
`so101_sim/env.py` 的 `EnvConfig.cycle_seconds`。

## 2. `meta/info.json` 字段

| 字段 | 本项目取值 | 说明 |
| --- | --- | --- |
| `codebase_version` | `"v3.0"` | 对齐 LeRobot v3 |
| `robot_type` | `"so101_sim"` | 仿真机器人（非真机） |
| `total_episodes` / `total_frames` / `total_tasks` | 80 / 4800 / 1（lang 为 3） | |
| `total_videos` | episodes × 2 | front + wrist |
| `chunks_size` | 1000 | 与官方一致；本项目全部落在 chunk-000 |
| `fps` / `dt` | 7.5 / 0.133333 | |
| `splits` | `{"train": "0:80"}` | |
| `data_path` | `data/chunk-{chunk:03d}/episode_{episode:06d}.parquet` | 与官方 v3 模板一致 |
| `video_path` | `videos/chunk-{chunk:03d}/{video_key}/episode_{episode:06d}.mp4` | 与官方 v3 模板一致 |
| `features` | 见下 | |
| `generation` | 仿真器 / 专家 / 域随机化列表 / `hardware_teleoperation: false` | **子集扩展字段**，明确声明非真机采集 |

`features`：

| key | dtype | shape | 说明 |
| --- | --- | --- | --- |
| `observation.state` | float32 | `[6]` | `[q1..q5, gripper]`，单位 rad / 归一化开合 |
| `action` | float32 | `[6]` | 绝对关节目标：`[q1..q5, gripper_target]` |
| `observation.images.front` | video | `[96,96,3]` | 前视相机（带安装误差抖动） |
| `observation.images.wrist` | video | `[96,96,3]` | 腕部俯视相机 |

## 3. parquet 列

每条 episode 一个 parquet，写入的列（`pa.FixedSizeListArray` 存向量）：

| 列 | 类型 | 含义 |
| --- | --- | --- |
| `observation.state` | fixed_size_list[float32, 6] | 观测关节状态 |
| `action` | fixed_size_list[float32, 6] | 动作目标 |
| `timestamp` | float32 | `frame_index / fps` |
| `frame_index` | int64 | episode 内帧号 0..59 |
| `episode_index` | int64 | episode 编号 |
| `index` | int64 | 数据集全局帧号 |
| `task_index` | int64 | 对应 `meta/tasks.jsonl` |
| `next.done` | bool | 每 episode 最后一帧为 true（对齐 v3 的 `next.done`） |

## 4. 视频

- 编码：H.264（`libx264`），`yuv420p`，`-crf 18`，`macro_block_size=1`（保持 96×96 不补齐）。
- 帧数 = episode 长度（60），解码后逐帧与 parquet 的 `frame_index` 对齐。
- 写入/读取使用 `imageio` + `imageio-ffmpeg`（自带 ffmpeg 二进制，无需系统安装）。

## 5. 与官方 LeRobot v3 的差异（重要）

| 项目 | 官方 LeRobot v3 | 本仓库 | 影响 |
| --- | --- | --- | --- |
| 包依赖 | 依赖 `lerobot` / `datasets` / `torchcodec` 等 | 只用 numpy / pyarrow / imageio | 轻量、可在 2 核 CPU 跑通；不能直接 `lerobot` 训练 |
| 数据文件 | v3 会把多个 episode 打包进 `file_000.parquet` 并通过 `data/chunk_index`、`file_index` 索引 | 每 episode 一个 `episode_XXXXXX.parquet` | 沿用 v2 风格文件粒度；`info.json` 的 `data_path` 仍是 v3 模板，读代码按 episode 编号定位 |
| 视频元数据 | `meta/episodes.jsonl` 含 `videos/<key>/chunk_index`、`file_index`、`from_timestamp` 等 | 只写 `length` / `tasks` / `dataset_from_index` / `dataset_to_index` | 官方工具若严格校验会缺字段 |
| 统计量 | `stats.json` 支持 `count/mean/std/min/max` 的完整形状 | state/action 为逐维 6 值；图像为逐通道 3 值 | 用于脚本内归一化；非官方完整格式 |
| 数据集划分 | 由用户自行定义 | `splits.train = "0:N"` | 训练脚本内部按最后 8 条 hold-out |
| 任务文本 | 任意 | 中文任务（ACT：`抓取积木并放进盒子`；lang：`把红色积木放进盒子里`） | 可直接验证中文语言条件 |
| 生成器 | 真机遥操作 / 公开数据集 | 仿真脚本专家 + 域随机化 | 见 `docs/real_robot.md`；README 不声称真机采集 |

`meta/episodes_meta.jsonl` 是**本仓库的扩展**：记录每条 episode 的目标颜色、
尺寸、位置、盒子位置、相机外参抖动、遮挡物、滑移概率等，便于失败复盘与
按场景切片统计；官方 v3 没有该文件。

## 6. 读写 API

```python
from so101_dataset import LeRobotDataset, LeRobotDatasetWriter

writer = LeRobotDatasetWriter("data/my_ds", fps=7.5, image_size=96)
writer.add_episode(front=frames_front, wrist=frames_wrist,
                   states=states, actions=actions,
                   task="抓取积木并放进盒子", scene_meta=scene_meta)
info = writer.finalize()

ds = LeRobotDataset.load("data/my_ds")
print(len(ds), ds.num_episodes, ds.fps)     # 4800 80 7.5
ep = ds.get_episode(0)                      # state/action/front/wrist/task/...
frame = ds.get_frame(1234)                  # 全局帧索引 -> 单帧
```

训练缓存：

```python
from so101_dataset import build_cache, load_cache
build_cache("data/demos_act", ".cache/act_cache.npz",
            image_size=64, chunk_size=20, val_episodes=8)
cache = load_cache(".cache/act_cache.npz")
# front/wrist (N,64,64,3) uint8, state (N,6), action_chunks (N,20,6),
# action_mask (N,20), episode_offsets (E+1,), meta=归一化统计 / 数据集信息
```

`build_action_chunks` 的语义：帧 `t` 的目标是 `action[t:t+chunk]`，在 episode
边界截断并用最后一个有效动作右填充；`action_mask` 标记有效位置，训练损失按
mask 归一化。

## 7. 如何验证

```bash
pytest tests/test_dataset.py -q          # roundtrip / stats / chunk / cache
python scripts/generate_dataset.py --out data/ci_mini --episodes 3   # 端到端
```

测试覆盖：`info.json` 关键字段、episodes/tasks/stats、parquet 与 mp4 的
roundtrip（state/action 精确一致，视频解码后与原始帧平均像素误差 < 12）、
全局帧索引、动作分块填充与 mask。
