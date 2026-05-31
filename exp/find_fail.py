"""Scan a rollouts directory for failed LIBERO episodes and write them to CSV.

Rollout videos are named like:
    <DATE_TIME>--episode=<N>--success=<bool>--task=<desc>.mp4
    <DATE_TIME>--with_future_img--episode=<N>--success=<bool>--task=<desc>.mp4

`episode=N` is the GLOBAL, 1-based episode counter across all tasks. With
`num_trials_per_task` trials per task (default 50), the task id and the
per-task episode index are:
    task_id       = (N - 1) // num_trials_per_task
    episode_idx   = (N - 1) %  num_trials_per_task

The CSV lists each failed episode once (ignoring the `with_future_img` copy),
so it can be fed back into the eval script to re-run those episodes in
planning mode.
"""

import argparse
import csv
import re
from pathlib import Path

FNAME_RE = re.compile(
    r"--episode=(?P<episode>\d+)--success=(?P<success>True|False)--task=(?P<task>.+)\.mp4$"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollouts_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "rollouts" / "2026_05_25",
        help="Directory containing rollout .mp4 files.",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path(__file__).resolve().parent / "fail_tasks.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--num_trials_per_task",
        type=int,
        default=50,
        help="Trials per task used during eval (for task_id / episode_idx mapping).",
    )
    args = parser.parse_args()

    fails = {}  # global_episode -> row
    for path in sorted(args.rollouts_dir.glob("*.mp4")):
        m = FNAME_RE.search(path.name)
        if not m:
            print(f"[skip] could not parse: {path.name}")
            continue
        if m.group("success") != "False":
            continue

        global_ep = int(m.group("episode"))
        if global_ep in fails:
            continue  # dedupe the with_future_img copy

        task_id = (global_ep - 1) // args.num_trials_per_task
        episode_idx = (global_ep - 1) % args.num_trials_per_task
        fails[global_ep] = {
            "task_id": task_id,
            "episode_idx": episode_idx,
            "global_episode": global_ep,
            "task": m.group("task"),
            "video": path.name,
        }

    rows = sorted(fails.values(), key=lambda r: (r["task_id"], r["episode_idx"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task_id", "episode_idx", "global_episode", "task", "video"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Found {len(rows)} failed episodes across "
          f"{len({r['task_id'] for r in rows})} tasks.")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
