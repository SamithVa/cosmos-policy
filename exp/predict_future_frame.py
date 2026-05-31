# Predict an action chunk + a future wrist frame from the fine-tuned cosmos_predict2_2b_480p_lerobot
# checkpoint on jokeru/record_p3_orange_1, starting at FRAME_IDX of episode 0.
#
# Outputs:
#   exp/predict_action_future_frame.mp4  — 16-frame side-by-side wrist video:
#       [ current frame (static) | GT future (t+1..t+chunk) | predicted future (static, @ t+chunk) ]
#   exp/predict_action_chunk.txt         — the predicted (chunk_size, 7) action chunk
#
# Notes:
#   * The policy predicts ONE future frame per camera (the observation at t+chunk_size),
#     not a 16-frame rollout — so the rightmost column is static, aligned with the last GT frame.
#   * Before running, consolidate your DCP checkpoint into a single .pt:
#       python -c "from torch.distributed.checkpoint.format_utils import dcp_to_torch_save; \
#                  dcp_to_torch_save('CHKPT_DIR/iter_000001000/model', 'CHKPT_DIR/iter_000001000/model.pt')"

import time
from types import SimpleNamespace

import numpy as np
import torch
import torchvision
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)

# ----------------------------------------------------------------------------- config
REPO_ID = "jokeru/record_p3_orange_1"
EPISODE = 0
FRAME_IDX = 50
CHUNK_SIZE = 16
N_CHUNKS = 20               # autoregressive queries; ~1 query advances chunk_size frames
NUM_DENOISING_STEPS = 5    # per-query denoise steps (higher = sharper future frame, slower)

CKPT_PATH = "./ckpt/orange_ckpt_10k.pt"
EXPERIMENT = "cosmos_predict2_2b_480p_lerobot"
CONFIG_FILE = "cosmos_policy/config/config.py"
T5_EMBEDDINGS_PATH = "/home/data/wanshan/.cache/cosmos_policy/lerobot/jokeru__record_p3_orange_1/t5_embeddings.pkl"
DATASET_STATS_PATH = "/home/data/wanshan/.cache/cosmos_policy/lerobot/jokeru__record_p3_orange_1/dataset_statistics.json"

OUT_VIDEO = "exp/predict_action_future_frame.mp4"
OUT_ACTIONS = "exp/predict_action_chunk.txt"
OUT_FPS = 16
SEP_PX = 4

# 3-camera mapping for our dataset (matches the "robocasa" suite branch in get_action):
# primary = top, secondary = down, wrist = wrist.
PRIMARY_KEY = "observation.images.top"
SECONDARY_KEY = "observation.images.down"
WRIST_KEY = "observation.images.wrist"
PROPRIO_KEY = "observation.state"


COMMON_HW = 224  # cameras have different native sizes; resize all to a common HxW before stacking


def chw_float_to_hwc_uint8(img: torch.Tensor) -> np.ndarray:
    """LeRobot (3, H, W) float [0,1] -> (COMMON_HW, COMMON_HW, 3) uint8 — common size so the
    eval pipeline's np.stack() over cameras doesn't choke on heterogeneous resolutions."""
    x = torch.nn.functional.interpolate(
        img.float().unsqueeze(0), size=(COMMON_HW, COMMON_HW), mode="bilinear", align_corners=False
    )[0]
    return (x * 255.0).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def main():
    # Minimal cfg covering every field get_model / get_action / prepare_images_for_model read.
    cfg = SimpleNamespace(
        config=EXPERIMENT,
        config_file=CONFIG_FILE,
        ckpt_path=CKPT_PATH,
        suite="robocasa",  # 3-camera path: wrist + primary + secondary
        use_wrist_image=True,
        num_wrist_images=1,
        use_third_person_image=True,
        num_third_person_images=2,
        use_proprio=True,
        normalize_proprio=True,
        chunk_size=CHUNK_SIZE,
        unnormalize_actions=True,
        use_variance_scale=False,
        trained_with_image_aug=True,
        use_jpeg_compression=False,
    )

    # 1) Load T5 cache from the precomputed pickle (avoids loading the 11B T5 encoder at runtime).
    init_t5_text_embeddings_cache(T5_EMBEDDINGS_PATH)

    # 2) Load the fine-tuned policy model.
    model, _ = get_model(cfg)
    dataset_stats = load_dataset_stats(DATASET_STATS_PATH)

    # 3) Read the conditioning frame + a window of GT future frames from episode 0.
    ds = LeRobotDataset(REPO_ID, revision="main", video_backend="pyav")
    cols = ds.hf_dataset.with_format("numpy")[:]
    mask = cols["episode_index"] == EPISODE
    ep_start = int(cols["index"][mask].min())
    ep_len = int(mask.sum())
    assert FRAME_IDX + CHUNK_SIZE < ep_len, f"episode {EPISODE} has {ep_len} frames; need {FRAME_IDX + CHUNK_SIZE + 1}"
    idx_to_task = {int(ti): str(t) for t, ti in zip(ds.meta.tasks.index, ds.meta.tasks["task_index"])}

    cur = ds[ep_start + FRAME_IDX]
    task = idx_to_task[int(cur["task_index"].item())]
    print(f"task: {task!r}")

    # Need enough GT frames to compare against the LAST predicted future (which is at
    # query_t + CHUNK_SIZE), so the dataset window is (N_CHUNKS-1)*CHUNK_SIZE for the
    # current timeline + CHUNK_SIZE more for the GT-future timeline.
    total_frames = N_CHUNKS * CHUNK_SIZE
    assert FRAME_IDX + total_frames + CHUNK_SIZE < ep_len, (
        f"need {FRAME_IDX + total_frames + CHUNK_SIZE} frames; episode has {ep_len}"
    )

    # 4) Real-data-driven rollout (mirrors run_libero_eval.py): every CHUNK_SIZE real frames,
    #    re-query the model with the REAL observation at that timestep. The "current" column in
    #    the video advances every frame; the predicted-future column updates only at each query
    #    and stays held for the next CHUNK_SIZE frames (one query → 16 future-action steps + one
    #    future-frame prediction). No AR drift, no proprio chicken-egg.
    def build_obs(ds_sample):
        return {
            "primary_image": chw_float_to_hwc_uint8(ds_sample[PRIMARY_KEY]),
            "secondary_image": chw_float_to_hwc_uint8(ds_sample[SECONDARY_KEY]),
            "wrist_image": chw_float_to_hwc_uint8(ds_sample[WRIST_KEY]),
            "proprio": ds_sample[PROPRIO_KEY].numpy().astype(np.float32),
        }

    # cameras as (row label, dataset key, predicted-frame key in get_action output)
    CAMERAS = [
        ("top",   PRIMARY_KEY,   "future_image"),
        ("down",  SECONDARY_KEY, "future_image2"),
        ("wrist", WRIST_KEY,     "future_wrist_image"),
    ]

    preds_per_chunk = []   # list of {cam_name: (H, W, 3) uint8}
    all_actions = []
    query_times = []        # seconds per get_action call (wall-clock, GPU-synced)
    for k in range(N_CHUNKS):
        query_t = ep_start + FRAME_IDX + k * CHUNK_SIZE
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        out = get_action(
            cfg=cfg,
            model=model,
            dataset_stats=dataset_stats,
            obs=build_obs(ds[query_t]),
            task_label_or_embedding=task,
            num_denoising_steps_action=NUM_DENOISING_STEPS,
            generate_future_state_and_value_in_parallel=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        query_times.append(elapsed)
        actions = np.asarray(out["actions"])  # (chunk_size, 7), unnormalized
        fut = out["future_image_predictions"]
        preds_per_chunk.append({name: fut[pred_key] for name, _, pred_key in CAMERAS})
        all_actions.append(actions)
        print(f"chunk {k + 1}/{N_CHUNKS} @ frame {FRAME_IDX + k * CHUNK_SIZE}: "
              f"value={out.get('value_prediction'):.4f}, time={elapsed * 1000:.0f} ms "
              f"({1.0 / elapsed:.2f} query/s, {CHUNK_SIZE / elapsed:.1f} action Hz), "
              f"last_action={actions[-1]}")

    # Steady-state timing summary (skip the first query — includes CUDA warm-up / kernel compile).
    qt = np.array(query_times)
    if len(qt) > 1:
        warm = qt[1:]
        print()
        print(f"== Inference timing summary (NUM_DENOISING_STEPS={NUM_DENOISING_STEPS}) ==")
        print(f"  first query (cold):           {qt[0] * 1000:.0f} ms")
        print(f"  steady mean  (n={len(warm)}):   {warm.mean() * 1000:.0f} ms  "
              f"-> {1.0 / warm.mean():.2f} query/s  -> {CHUNK_SIZE / warm.mean():.1f} action Hz")
        print(f"  steady min:                   {warm.min() * 1000:.0f} ms  -> {CHUNK_SIZE / warm.min():.1f} action Hz")
        print(f"  steady max:                   {warm.max() * 1000:.0f} ms  -> {CHUNK_SIZE / warm.max():.1f} action Hz")
        print("  ('action Hz' = chunk_size / query_time, i.e. effective rate if you execute")
        print("   the full chunk open-loop before re-querying — matches num_open_loop_steps=chunk_size.)")

    all_actions = np.concatenate(all_actions, axis=0)  # (N_CHUNKS * chunk_size, 7)
    np.savetxt(OUT_ACTIONS, all_actions, fmt="%.6f",
               header=f"predicted action chunks concatenated ({N_CHUNKS}*{CHUNK_SIZE}, 7)")
    print(f"saved actions -> {OUT_ACTIONS}  ({all_actions.shape})")

    # 5) Build a 3-row × 3-column grid video. Each row is a camera (top / down / wrist);
    #    each row holds [current real t | GT @ query+chunk | predicted @ query+chunk].
    #    Cameras have different native aspect ratios → resize all panels to the predicted
    #    frame size (224×224) so rows line up.
    target_h, target_w = preds_per_chunk[0]["wrist"].shape[:2]

    def resize_chw_to_hwc_uint8(t: torch.Tensor) -> np.ndarray:
        x = t.float().unsqueeze(0)  # (1, 3, H, W) in [0,1]
        x = torch.nn.functional.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return (x[0] * 255.0).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()

    h_sep = np.full((target_h, SEP_PX, 3), 255, dtype=np.uint8)                 # between columns
    v_sep = np.full((SEP_PX, 3 * target_w + 2 * SEP_PX, 3), 255, dtype=np.uint8)  # between rows

    # Precompute per-camera GT future frame at each query point (held for CHUNK_SIZE frames).
    gt_per_chunk = {
        name: [
            resize_chw_to_hwc_uint8(ds[ep_start + FRAME_IDX + k * CHUNK_SIZE + CHUNK_SIZE][ds_key])
            for k in range(N_CHUNKS)
        ]
        for name, ds_key, _ in CAMERAS
    }

    frames = []
    for i in range(total_frames):
        # one (H, 3W+2sep, 3) row per camera
        rows = []
        for name, ds_key, _ in CAMERAS:
            current_hwc = resize_chw_to_hwc_uint8(ds[ep_start + FRAME_IDX + i][ds_key])
            gt_hwc = gt_per_chunk[name][i // CHUNK_SIZE]
            pred_hwc = preds_per_chunk[i // CHUNK_SIZE][name]
            rows.append(np.concatenate([current_hwc, h_sep, gt_hwc, h_sep, pred_hwc], axis=1))
        # stack rows vertically with separators
        stacked = rows[0]
        for r in rows[1:]:
            stacked = np.concatenate([stacked, v_sep, r], axis=0)
        frames.append(stacked)
    video = np.stack(frames, axis=0)
    torchvision.io.write_video(OUT_VIDEO, torch.from_numpy(video), fps=OUT_FPS)
    duration_s = total_frames / OUT_FPS
    print(f"saved {video.shape[0]}-frame {video.shape[1]}x{video.shape[2]} 3-cam grid "
          f"({duration_s:.1f} s @ {OUT_FPS} fps) -> {OUT_VIDEO}")
    print("rows: [top, down, wrist]    columns: [current real t | GT real t+chunk | predicted t+chunk]")


if __name__ == "__main__":
    main()
