#!/usr/bin/env python3
"""Decode a LIBERO-plus rollout episode / task_id into its task name + perturbation.

Usage:
    # from a rollout filename (extracts episode=NNN)
    python exp/decode_task.py rollouts/2026_05_27/..._episode=34251_....mp4

    # from a global episode number (video_idx)
    python exp/decode_task.py 34251

    # from a suite-local task_id directly
    python exp/decode_task.py 685 --task-id

Options:
    --suite libero_10        Task suite the run used (default: libero_10)
    --ntpt 50                num_trials_per_task used at eval time (default: 50).
                             Rollout video_idx = task_id * ntpt + episode_idx + 1,
                             so this MUST match the run (LIBERO-plus runs often use 1).
    --task-id                Treat the input number as a task_id, not an episode number.

Reads LIBERO-plus's task_classification.json (no torch / sim import needed), so it is
fast and dependency-light. task_id (0-indexed) == classification "id" - 1.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CLS_JSON = os.path.join(
    HERE, "..", "LIBERO-plus", "libero", "libero", "benchmark", "task_classification.json"
)


def decode_perturbation(name: str) -> list[str]:
    """Return human-readable lines describing the perturbation encoded in a task name."""
    lines = []

    # Camera viewpoint: ..._view_<horiz>_<vert>_<scale>_<endRot>_<endVert>[_initstate_N][_noise_N]
    m = re.search(r"_view_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)", name)
    if m:
        h, v, s, er, ev = (int(x) for x in m.groups())
        scale = s / 100.0
        parts = []
        if h or v or scale != 1.0:
            parts.append(
                f"camera POSITION moved (orbit horiz={h}°, vert={v}°, zoom scale={scale})"
            )
        if er or ev:
            # end_point_* rotate orientation in place; values are degrees mod 360
            er_s = er if er <= 180 else er - 360
            ev_s = ev if ev <= 180 else ev - 360
            parts.append(f"camera AIM rotated in place (yaw={er_s}°, pitch={ev_s}°)")
        if not parts:
            parts.append("camera at default (no change)")
        lines.append("camera: " + "; ".join(parts))

    mi = re.search(r"_initstate_(\d+)", name)
    if mi:
        n = int(mi.group(1))
        lines.append(
            f"robot init state variant = {n}" + (" (default pose)" if n == 0 else " (perturbed robot pose)")
        )
    mn = re.search(r"_noise_(\d+)", name)
    if mn:
        lines.append(f"sensor noise level = {int(mn.group(1))}")
    if re.search(r"_table_\d+|_tb_\d+", name):
        mt = re.search(r"_t(?:able|b)_(\d+)", name)
        lines.append(f"background/table texture variant = {mt.group(1) if mt else '?'}")
    if "_light_" in name:
        ml = re.search(r"_light_(\d+)", name)
        lines.append(f"light condition variant = {ml.group(1) if ml else '?'}")
    if "_add_" in name:
        ma = re.search(r"_add_(\d+)", name)
        lines.append(f"added confounding object(s), layout variant = {ma.group(1) if ma else '?'}")
    if "_language_" in name:
        lines.append("language instruction rewritten (see BDDL for exact text)")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="rollout filename, global episode number, or task_id (with --task-id)")
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--ntpt", type=int, default=50, help="num_trials_per_task used at eval (default 50)")
    ap.add_argument("--task-id", action="store_true", help="treat target as a task_id")
    args = ap.parse_args()

    # Resolve the target to (task_id, episode_in_task)
    episode_in_task = 0
    if args.task_id:
        task_id = int(args.target)
    else:
        m = re.search(r"episode=(\d+)", args.target)
        video_idx = int(m.group(1)) if m else int(args.target)
        task_id = (video_idx - 1) // args.ntpt
        episode_in_task = (video_idx - 1) % args.ntpt

    cls = json.load(open(CLS_JSON))
    if args.suite not in cls:
        sys.exit(f"suite {args.suite!r} not in classification; available: {list(cls)}")
    items = cls[args.suite]
    if not (0 <= task_id < len(items)):
        sys.exit(f"task_id {task_id} out of range for {args.suite} (0..{len(items)-1})")

    item = items[task_id]  # classification id == task_id + 1, list is in task order
    assert item["id"] == task_id + 1, f"ordering mismatch: id={item['id']} task_id={task_id}"

    print(f"suite            : {args.suite}")
    print(f"task_id          : {task_id}   (episode_in_task={episode_in_task}, ntpt={args.ntpt})")
    print(f"name             : {item['name']}")
    print(f"category         : {item['category']}")
    print(f"difficulty_level : {item.get('difficulty_level')}")
    print("perturbation     :")
    for line in decode_perturbation(item["name"]) or ["(none decoded)"]:
        print(f"  - {line}")


if __name__ == "__main__":
    main()
