# Minimal real-robot control loop for the fine-tuned cosmos_predict2_2b_480p_lerobot
# "orange" checkpoint (3-camera robocasa layout: wrist + top(primary) + down(secondary)).
#
# This is the deployment counterpart of exp/predict_future_frame.py: same model load and
# same get_action() call, but the observation source is the LIVE robot instead of a recorded
# LeRobot dataset, and the predicted action chunk is executed on the robot.
#
# It runs the policy locally (no HTTP server) — simplest when the policy GPU box also talks
# to the robot. To serve over the network instead, adapt aloha/deploy.py to suite="robocasa".
#
# YOU MUST FILL IN TWO HARDWARE HOOKS (search for "TODO[robot]"):
#   1) get_observation()  -> grab the 3 live camera frames + 7-dim proprio
#   2) send_action(a)     -> command one 7-dim action (6 DoF + gripper) on the robot
#
# Before running, consolidate the DCP checkpoint into a single .pt (see predict_future_frame.py).

import time
from types import SimpleNamespace

import numpy as np
import torch

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)

# ----------------------------------------------------------------------------- config
CKPT_PATH = "./ckpt/orange_ckpt_10k.pt"
EXPERIMENT = "cosmos_predict2_2b_480p_lerobot"
CONFIG_FILE = "cosmos_policy/config/config.py"
T5_EMBEDDINGS_PATH = "/home/data/wanshan/.cache/cosmos_policy/lerobot/jokeru__record_p3_orange_1/t5_embeddings.pkl"
DATASET_STATS_PATH = "/home/data/wanshan/.cache/cosmos_policy/lerobot/jokeru__record_p3_orange_1/dataset_statistics.json"

# MUST be an exact key in the T5 pickle, or the 11B T5 encoder is loaded on-the-fly (slow).
TASK = "Pick up the orange and put it into the basket."

CHUNK_SIZE = 16              # model predicts this many actions per query
NUM_OPEN_LOOP_STEPS = 8      # how many of the chunk to execute before re-querying (<= CHUNK_SIZE)
NUM_DENOISING_STEPS = 1      # per-query denoise steps (matches predict_future_frame.py)
CONTROL_HZ = 16              # rate at which individual actions are sent to the robot
MAX_STEPS = 10_000           # safety cap on total commanded actions

COMMON_HW = 224              # cameras resized to this square before stacking (matches training prep)


def hwc_uint8(img: np.ndarray) -> np.ndarray:
    """Resize an arbitrary HxWx3 uint8 camera frame to (COMMON_HW, COMMON_HW, 3) uint8.

    get_action() expects HWC uint8 images; resizing here mirrors the preprocessing the
    offline predict script does so the model sees the same input distribution."""
    x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # (1,3,H,W)
    x = torch.nn.functional.interpolate(x, size=(COMMON_HW, COMMON_HW), mode="bilinear", align_corners=False)
    return (x[0] * 255.0).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()


# ----------------------------------------------------------------------------- robot hooks
def get_observation() -> dict:
    """TODO[robot]: return ONE live observation from your robot.

    Required keys (camera assignment MUST match training — see predict_future_frame.py:52-54):
      - "primary_image":   top camera   (H, W, 3) uint8
      - "secondary_image": down camera  (H, W, 3) uint8
      - "wrist_image":     wrist camera (H, W, 3) uint8
      - "proprio":         (7,) float32 robot state, same units/order as the training dataset

    Replace the raise below with calls into your camera + robot SDK, e.g.:
        top   = camera_top.read()       # HxWx3 uint8 BGR/RGB — convert to RGB to match training
        down  = camera_down.read()
        wrist = camera_wrist.read()
        state = robot.get_proprio()      # 6 DoF pose + 1 gripper -> (7,) float32
        return {
            "primary_image":   hwc_uint8(top),
            "secondary_image": hwc_uint8(down),
            "wrist_image":     hwc_uint8(wrist),
            "proprio":         np.asarray(state, dtype=np.float32),
        }
    """
    raise NotImplementedError("Fill in get_observation() with your camera + robot SDK calls.")


def send_action(action: np.ndarray) -> None:
    """TODO[robot]: command one 7-dim action on the robot.

    `action` is a (7,) float32, ALREADY UNNORMALIZED (get_action unnormalizes via dataset_stats).
    Layout matches your training action vector — typically 6 DoF (e.g. delta or abs pose) + 1 gripper.
    Map it to your controller, e.g.:
        robot.set_eef_delta(action[:6]); robot.set_gripper(action[6])
    """
    raise NotImplementedError("Fill in send_action() to drive your robot controller.")


def main():
    # Minimal cfg covering every field get_model / get_action / prepare_images_for_model read.
    # Identical to predict_future_frame.py so the deployed model behaves like the validated one.
    cfg = SimpleNamespace(
        config=EXPERIMENT,
        config_file=CONFIG_FILE,
        ckpt_path=CKPT_PATH,
        suite="robocasa",            # 3-camera path: wrist + primary(top) + secondary(down)
        use_wrist_image=True,
        num_wrist_images=1,
        use_third_person_image=True,
        num_third_person_images=2,
        use_proprio=True,
        normalize_proprio=True,
        chunk_size=CHUNK_SIZE,
        unnormalize_actions=True,
        use_variance_scale=False,
        trained_with_image_aug=True,  # must match how the model was trained
        use_jpeg_compression=False,
    )

    # 1) T5 cache (avoids loading the 11B encoder at runtime).
    init_t5_text_embeddings_cache(T5_EMBEDDINGS_PATH)
    # 2) Policy model + dataset stats (for proprio normalize / action unnormalize).
    model, _ = get_model(cfg)
    dataset_stats = load_dataset_stats(DATASET_STATS_PATH)

    dt = 1.0 / CONTROL_HZ
    steps = 0
    print(f"Deploying. task={TASK!r}  chunk={CHUNK_SIZE} open_loop={NUM_OPEN_LOOP_STEPS} @ {CONTROL_HZ} Hz")
    print("Ctrl-C to stop.")
    try:
        while steps < MAX_STEPS:
            # Re-query the policy with the CURRENT live observation.
            obs = get_observation()
            t0 = time.perf_counter()
            out = get_action(
                cfg=cfg,
                model=model,
                dataset_stats=dataset_stats,
                obs=obs,
                task_label_or_embedding=TASK,
                num_denoising_steps_action=NUM_DENOISING_STEPS,
                generate_future_state_and_value_in_parallel=False,  # actions only — deployment fast path
            )
            actions = np.asarray(out["actions"], dtype=np.float32)  # (CHUNK_SIZE, 7), unnormalized
            print(f"query {steps // NUM_OPEN_LOOP_STEPS}: {(time.perf_counter() - t0) * 1000:.0f} ms")

            # Execute the first NUM_OPEN_LOOP_STEPS of the chunk open-loop, then re-query.
            for a in actions[:NUM_OPEN_LOOP_STEPS]:
                loop_start = time.perf_counter()
                send_action(a)
                steps += 1
                if steps >= MAX_STEPS:
                    break
                sleep = dt - (time.perf_counter() - loop_start)
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        print(f"\nStopped after {steps} actions.")


if __name__ == "__main__":
    main()
