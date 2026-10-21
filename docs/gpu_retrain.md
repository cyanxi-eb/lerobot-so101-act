# GPU 重训实验：更大的 ACT、数据增广与双模型集成

CPU 版训练预算是 65 分钟、0.68 M 参数、64×64 输入，成功率明显被算力卡住
（README §8）。这份文档记录 2026-10-21 在一张 32 GB 单卡上的重训实验：
240 条演示、96×96 输入、两种 ACT 配置、同协议 20 条 + 50 条两套评测。

> 边界不变：仍然是仿真、脚本专家数据、几何规则判定成功；没有真机，也没有
> 在 ROS2 上实跑。这里想验证的是“换 GPU 之后训练预算花在哪儿最值”。

## 1. 机器与预算

| 项目 | 值 |
| --- | --- |
| 算力 | 单卡 vGPU-32GB（32 GB 显存，约 RTX 4090 级） |
| 驱动 / CUDA | 550.120 / 12.4 |
| 框架 | PyTorch 2.5.1+cu124 |
| CPU / 内存 | 12 vCPU / 96 GB |
| 计费 | ¥1.66/时，按量 |
| 实际耗时 | 约 33 分钟（含 240 条数据生成、两轮训练、四组评测、上下行） |
| 费用 | 约 ¥0.9 |

## 2. 数据

```bash
python scripts/generate_dataset.py --out data/demos_act_240 --episodes 240 \
  --image-size 96 --seed 0 --report reports/gpu/dataset_act_240.json
```

240 条 episode / 14,400 帧 / 7.9 MB，脚本专家成功率 96.8%，生成耗时 138 s
（12 核机器；CPU 版 80 条曾要 137 s）。数据在实验机器上生成，不随仓库分发；
report 里保留了每条 episode 的 seed 和随机化参数。

## 3. 训练

两条路线，共用同一份 240 条数据、96×96 输入、chunk=20、batch=128：

| 配置 | 参数量 | epoch | 关键超参 | 训练耗时 | 选中的 epoch |
| --- | ---: | ---: | --- | ---: | ---: |
| `act_240_gpu.pt`（大模型） | 2.09 M | 150 | d_model 192 / 3+3 层 / CNN 48 | **10.2 min** | 70 |
| `act_240_aug.pt`（原架构 + 增广） | 0.68 M | 100 | d_model 128 / 2+2 层 / CNN 32 + 光度/平移增广 | **7.3 min** | 30 |

训练脚本新增了 `--device auto|cpu|cuda`、`--d-model/--n-heads/--n-layers/--cnn-width`
和 `--augment`，CPU 路径和默认参数保持原样（CI 不受影响）。增广实现在
`so101_policy.training.augment_batch`：逐通道颜色增益、亮度/对比度、±3 px
平移、σ=0.015 高斯噪声，只在训练 batch 上生效。

```bash
# 大模型
python scripts/train_act.py --dataset data/demos_act_240 \
  --image-size 96 --chunk-size 20 --batch-size 128 --epochs 150 \
  --max-minutes 16 --eval-every 10 --eval-trials 10 --lr 3e-4 \
  --d-model 192 --n-heads 8 --n-layers 3 --cnn-width 48 --device cuda

# 原架构 + 增广
python scripts/train_act.py --dataset data/demos_act_240 \
  --image-size 96 --chunk-size 20 --batch-size 128 --epochs 100 \
  --max-minutes 12 --eval-every 10 --eval-trials 12 --lr 3e-4 \
  --augment --device cuda
```

训练吞吐约 25 steps/s（GPU，batch=128），单个 epoch 3.4–4.5 s；两个 checkpoint
都按闭环 rollout 成功率选最优，所以文件里的 epoch 不是最后一个。

## 4. 评测（同协议、同 seed）

协议与 CPU 版一致：96×96 双相机、replan=5、无时序集成、成功 = 积木落入盒内。
所有策略都重跑同一批 seed，避免拿旧的 CPU 数字和新的 GPU 数字直接比。

### 4.1 20 trials / split（与 CPU 版同一套 seed）

| 策略 | seen | unseen | chunk 延迟 mean（seen / unseen） |
| --- | ---: | ---: | ---: |
| CPU baseline（80 条，0.68 M，原 checkpoint，同协议重跑） | 8/20（40%） | 3/20（15%） | 4.1 / 1.6 ms |
| GPU 大模型（240 条，2.09 M） | 10/20（50%） | 1/20（5%） | 4.2 / 1.8 ms |
| GPU 增广（240 条，0.68 M + aug） | 8/20（40%） | **7/20（35%）** | 4.2 / 3.2 ms |
| **双 checkpoint 集成**（大模型 + 增广，等权平均 chunk） | **11/20（55%）** | 5/20（25%） | 6.5 / 3.4 ms |

读法：

- **增广是未见分布的关键**：unseen 15% → 35%，说明颜色/亮度/平移扰动让策略
  不再死记训练时的颜色统计；
- **大模型对已见分布更有利**（40% → 50%），但几乎不迁移到未见颜色（5%）；
- 把两者按 chunk 等权平均，seen 55%、unseen 25%，两个 split 都不低于原基线；
- 集成让 chunk 延迟翻倍（6.5 ms），仍然远低于 7.5 Hz 控制周期（133 ms）。

### 4.2 50 trials / split 复评

| 策略 | seen | unseen |
| --- | ---: | ---: |
| CPU baseline（80 条） | 19/50（38%） | 4/50（8%） |
| 双 checkpoint 集成 | 未跑（本地 CPU 复评超时） | 未跑 |

50 trials 把二项分布的标准误从 ~11% 压到 ~7%。基线从 20 trials 的 40% / 15%
落到 38% / 8%，说明 20 trials 的 unseen 数字本身波动就不小；集成在 20 trials
上的 55% / 25% 应该按同样的区间打折看，不宜当作精确值。50 trials 的集成复评
留到下一轮（脚本已就绪：`scripts/eval_ensemble.py --seen-trials 50 --unseen-trials 50`）。

## 5. 限制与下一步

1. **失败仍以 grasp_miss 为主**（集成在 20 trials 上 9/9 失败都是抓空），
   说明视觉定位精度仍然是瓶颈，不是“抓到了放偏”；
2. 集成推理要跑两个模型，延迟 2×；如果上真机，优先考虑蒸馏成单模型；
3. 240 条仍是小数据；下一步把演示加到 400–600 条，并加运动模糊/曝光增广；
4. 动作空间改成「当前位姿 + delta」冲 cm 级精度；
5. 语言条件策略这次没有重训（仍是 CPU 版 15%），它的 GPU 训练命令和 ACT 同构，
   下一步用 160 条 lang 数据 + 同样增广重跑；
6. 20/50 trials 都是仿真 seed，置信区间见 docs/evaluation.md；没有真机结论。

## 6. 产物

- `checkpoints/act_240_gpu.pt`（8.4 MB，2.09 M 参数）
- `checkpoints/act_240_aug.pt`（2.8 MB，0.68 M 参数）
- `reports/gpu/`：数据、训练曲线、四组 eval 的 JSON + Markdown + 失败画面
- `scripts/eval_ensemble.py`：双 checkpoint 集成评测（`--checkpoint-a/--checkpoint-b`）
