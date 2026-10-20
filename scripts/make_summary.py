#!/usr/bin/env python
"""Aggregate training / evaluation / dataset reports into one summary table.

Reads the JSON reports written by the other scripts and emits
``reports/summary.md`` + ``reports/summary.json``.  Used to fill the README
metrics table without hand-copying numbers.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"


def read(name: str) -> dict | None:
    path = REPORTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def split_row(name: str, split: dict | None) -> str:
    if not split:
        return f"| {name} | - | - | - | - | - | - | - |"
    failures = ", ".join(f"{k}:{v}" for k, v in split["failures"].items()) or "-"
    return (
        f"| {name} | {split['trials']} | {split['successes']} | "
        f"{split['success_rate'] * 100:.1f}% | "
        f"{split['latency_ms_mean']:.1f} / {split['latency_ms_p95']:.1f} | "
        f"{split['cycle_seconds_mean']:.2f} | "
        f"{split.get('wall_seconds_mean', float('nan')):.2f} | {failures} |"
    )


def main() -> int:
    dataset_act = read("dataset_act.json")
    dataset_lang = read("dataset_lang.json")
    train_act = read("train_act.json")
    train_lang = read("train_lang.json")
    eval_act = read("eval_act.json")
    eval_lang = read("eval_lang.json")

    summary: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "device": "CPU only (2 physical cores / 4 logical threads)",
            "torch_threads": (train_act or {}).get("torch_threads"),
            "simulator": "so101_sim (kinematic, no physics engine)",
            "image_size_dataset": 96,
            "image_size_policy": 64,
        },
        "datasets": {
            "act": None if not dataset_act else {
                k: dataset_act[k] for k in (
                    "episodes", "frames", "episode_length", "fps",
                    "cycle_seconds", "target_color_counts",
                    "expert_success_rate", "dataset_mb", "generation_seconds")
            },
            "lang": None if not dataset_lang else {
                k: dataset_lang[k] for k in (
                    "episodes", "frames", "episode_length", "fps",
                    "cycle_seconds", "target_color_counts",
                    "expert_success_rate", "dataset_mb", "generation_seconds")
            },
        },
        "act_training": None if not train_act else {
            k: train_act[k] for k in (
                "train_minutes", "epochs_done", "best_epoch", "best_success_rate",
                "best_val_l1", "parameter_count", "checkpoint_bytes",
                "steps_per_second", "total_steps")
        },
        "lang_training": None if not train_lang else {
            k: train_lang[k] for k in (
                "train_minutes", "epochs_done", "best_epoch", "best_success_rate",
                "best_val_l1", "parameter_count", "checkpoint_bytes",
                "steps_per_second", "total_steps", "vocab_size")
        },
        "act_eval": None if not eval_act else {
            "checkpoint_bytes": eval_act["checkpoint"]["bytes"],
            "n_parameters": eval_act["checkpoint"]["n_parameters"],
            "replan_interval": eval_act["protocol"]["replan_interval"],
            "splits": eval_act["splits"],
            "failure_mode_counts": eval_act.get("failure_mode_counts"),
            "failure_cases": eval_act.get("failure_cases", []),
        },
        "lang_eval": None if not eval_lang else {
            "checkpoint_bytes": eval_lang["checkpoint"]["bytes"],
            "n_parameters": eval_lang["checkpoint"]["n_parameters"],
            "replan_interval": eval_lang["protocol"]["replan_interval"],
            "splits": eval_lang["splits"],
            "failure_mode_counts": eval_lang.get("failure_mode_counts"),
            "failure_cases": eval_lang.get("failure_cases", []),
        },
    }

    lines = [
        "# 实测结果汇总",
        "",
        f"- 生成时间：{summary['generated_at']}",
        f"- 环境：{summary['environment']['device']}，"
        f"torch threads={summary['environment']['torch_threads']}",
        "- 全部为仿真闭环 rollout 实测值；**无真机实验**。",
        "",
        "## 数据集",
        "",
        "| 数据集 | episodes | 帧数 | 帧/条 | fps | 单次循环 | 专家成功率 | 磁盘 | 生成耗时 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (("act", "demos_act"), ("lang", "demos_lang")):
        ds = summary["datasets"][key]
        if ds:
            lines.append(
                f"| {label} | {ds['episodes']} | {ds['frames']} | "
                f"{ds['episode_length']} | {ds['fps']} | {ds['cycle_seconds']} s | "
                f"{ds['expert_success_rate'] * 100:.1f}% | {ds['dataset_mb']} MB | "
                f"{ds['generation_seconds']:.0f} s |"
            )
    lines += [
        "",
        "## 训练",
        "",
        "| 模型 | 参数量 | 训练耗时 | epochs | checkpoint | 最终/最优 rollout 成功率（训练期验证） |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (("act_training", "ACT"), ("lang_training", "Lang-cond")):
        tr = summary[key]
        if tr:
            lines.append(
                f"| {label} | {tr['parameter_count'] / 1e6:.2f} M | "
                f"{tr['train_minutes']:.1f} min | {tr['epochs_done']} | "
                f"{tr['checkpoint_bytes'] / 1e6:.2f} MB | "
                f"{tr['best_success_rate'] * 100:.1f}% (epoch {tr['best_epoch']}) |"
            )
    lines += [
        "",
        "> ACT 训练因 CPU 吞吐受限分 3 段续训，合计约 65 min；上表 20.3 min 是",
        "> 最后一段（10 epoch）的耗时，前两段见 `reports/train_act.json` 的说明与",
        "> `docs/evaluation.md` §3.3。语言条件策略为单段 18.0 min。",
    ]
    lines += [
        "",
        "## 评测（真实闭环，20 trials / split）",
        "",
        "| split | trials | 成功 | 成功率 | chunk 延迟 mean/p95 (ms) | 仿真循环 (s) | "
        "wall 循环 (s) | 失败模式 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if summary["act_eval"]:
        for name, split in summary["act_eval"]["splits"].items():
            lines.append(split_row(f"ACT {name}", split))
    if summary["lang_eval"]:
        for name, split in summary["lang_eval"]["splits"].items():
            lines.append(split_row(f"Lang-cond {name}", split))
        lang = summary["lang_eval"]["splits"].get("lang", {})
        lines += [
            "",
            f"- 语言策略指令遵循（完整任务）成功率：{lang.get('instruction_accuracy', 0) * 100:.1f}%",
            f"- 放错积木率：{lang.get('wrong_object_rate', 0) * 100:.1f}%",
            f"- 抓取 grounding（抓到指令指定积木 / 发生抓取的 trial）："
            f"{lang.get('grounded_grasps', 0)}/{lang.get('grasp_trials', 0)}"
            + (f" = {lang['grounding_accuracy'] * 100:.1f}%"
               if lang.get("grounding_accuracy") is not None else ""),
        ]

    lines += ["", "## 失败案例（自动摘录）", ""]
    for label, payload in (("ACT", summary["act_eval"]), ("Lang", summary["lang_eval"])):
        if not payload:
            continue
        for case in payload.get("failure_cases", [])[:4]:
            lines.append(f"- {label}: {case['description']}")
    lines += [
        "",
        "## 报告文件",
        "",
        "- `reports/eval_act.md` / `reports/eval_lang.md`：完整评测报告（含失败拼图）",
        "- `reports/train_act_finetune.json` / `reports/train_lang.json`：训练日志",
        "- `reports/dataset_act.json` / `reports/dataset_lang.json`：数据集统计",
        "- `reports/eval_act_failures.png`：失败案例画面",
        "",
    ]

    (REPORTS / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
