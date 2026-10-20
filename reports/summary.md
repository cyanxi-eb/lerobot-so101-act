# 实测结果汇总

- 环境：CPU（无 GPU）。
- 全部为仿真闭环 rollout 实测值；无真机实验。

## 数据集

| 数据集 | episodes | 帧数 | 帧/条 | fps | 单次循环 | 专家成功率 | 磁盘 | 生成耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| demos_act | 80 | 4800 | 60 | 7.5 | 8.0 s | 98.8% | 2.64 MB | 137 s |
| demos_lang | 80 | 4800 | 60 | 7.5 | 8.0 s | 98.8% | 2.93 MB | 165 s |

## 训练

| 模型 | 参数量 | 训练耗时 | epochs | checkpoint | 最终/最优 rollout 成功率（训练期验证） |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACT | 0.68 M | 20.3 min | 10 | 2.76 MB | 50.0% (epoch 8) |
| Lang-cond | 0.41 M | 18.0 min | 10 | 1.66 MB | 16.7% (epoch 9) |

> ACT 训练因 CPU 吞吐受限分 3 段续训，合计约 65 min；上表 20.3 min 是
> 最后一段（10 epoch）的耗时，前两段见 `reports/train_act.json` 的说明与
> `docs/evaluation.md` §3.3。语言条件策略为单段 18.0 min。

## 评测（真实闭环，20 trials / split）

| split | trials | 成功 | 成功率 | chunk 延迟 mean/p95 (ms) | 仿真循环 (s) | wall 循环 (s) | 失败模式 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACT seen | 20 | 10 | 50.0% | 15.3 / 38.8 | 7.41 | 0.96 | slip:1, grasp_miss:9 |
| ACT unseen | 20 | 2 | 10.0% | 12.4 / 24.5 | 7.93 | 0.83 | grasp_miss:18 |
| Lang-cond lang | 20 | 3 | 15.0% | 20.9 / 51.4 | 7.87 | 1.27 | grasp_miss:16, slip:1 |

- 语言策略指令遵循（完整任务）成功率：15.0%
- 放错积木率：10.0%
- 抓取 grounding（抓到指令指定积木 / 发生抓取的 trial）：4/4 = 100.0%

## 失败案例（自动摘录）

- ACT: seed 3000, failure=slip, target=red (half-size 0.0178 m), occluder=False, steps=60
- ACT: seed 3001, failure=grasp_miss, target=red (half-size 0.0212 m), occluder=True, steps=60
- ACT: seed 3002, failure=grasp_miss, target=blue (half-size 0.0199 m), occluder=True, steps=60
- ACT: seed 3004, failure=grasp_miss, target=red (half-size 0.0217 m), occluder=True, steps=60
- Lang: seed 3200, failure=grasp_miss, target=red (half-size 0.0203 m), occluder=False, steps=60, a distractor block ended up in the box
- Lang: seed 3202, failure=grasp_miss, target=green (half-size 0.0176 m), occluder=False, steps=60
- Lang: seed 3203, failure=grasp_miss, target=red (half-size 0.0186 m), occluder=False, steps=60
- Lang: seed 3204, failure=grasp_miss, target=blue (half-size 0.0213 m), occluder=False, steps=60

## 报告文件

- `reports/eval_act.md` / `reports/eval_lang.md`：完整评测报告（含失败拼图）
- `reports/train_act.json` / `reports/train_lang.json`：训练日志
- `reports/dataset_act.json` / `reports/dataset_lang.json`：数据集统计
- `reports/eval_act_failures.png`：失败案例画面
