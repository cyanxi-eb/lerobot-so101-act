# Evaluation - act policy

- Checkpoint: `checkpoints/act_240_aug.pt` (2.77 MB, 0.68M parameters)
- Control: 7.5 Hz, max 60 steps (8.0 s), replan every 5 steps
- Success: the instructed block ends inside the box after release

| split | trials | success | success rate | chunk latency (mean / p95) | sim cycle | wall cycle | failure modes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| seen | 20 | 8 | 40.0% | 4.176 / 5.378 ms | 7.513 s | 0.217 s | slip: 1, grasp_miss: 11 |
| unseen | 20 | 7 | 35.0% | 3.158 / 3.845 ms | 7.687 s | 0.272 s | grasp_miss: 11, timeout: 2 |

## Failure cases

- seed 3000, failure=slip, target=red (half-size 0.0178 m), occluder=False, steps=60
- seed 3001, failure=grasp_miss, target=red (half-size 0.0212 m), occluder=True, steps=60
- seed 3003, failure=grasp_miss, target=blue (half-size 0.0194 m), occluder=True, steps=60
- seed 3004, failure=grasp_miss, target=red (half-size 0.0217 m), occluder=True, steps=60
- seed 3006, failure=grasp_miss, target=green (half-size 0.0206 m), occluder=False, steps=60

![failure cases](eval_act_failures.png)

## Notes

- Simulator-only evaluation. There is no real SO-101 hardware in this project;
  see `docs/real_robot.md` for the real-robot protocol.
- Scene seeds are disjoint from training and from the seeds used for
  training-time checkpoint selection.
- `unseen` changes colour (yellow/purple never seen in training), block size
  and workspace position, so the gap to `seen` is a real distribution-shift gap.
- Latency is measured per action chunk on this CPU (batch size 1), including
  image resize / normalisation but excluding simulator rendering.
