# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""
LeRobot (Hugging Face Hub) dataloader for Cosmos Policy fine-tuning.

Emits the same sample dict as the RoboCasa dataloader (11-segment latent layout:
blank, proprio, wrist, primary, secondary, action, future proprio, future wrist,
future primary, future secondary, value), so it plugs into the RoboCasa-style
model config (state_t=11, chunk_duration=41).

Run this command to print a few samples from a LeRobot dataset:
    python -m cosmos_policy.datasets.lerobot_dataset
"""

import os
import pickle

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_policy.datasets.dataset_common import (
    compute_monte_carlo_returns,
    get_action_chunk_with_padding,
    load_or_compute_dataset_statistics,
    load_or_compute_post_normalization_statistics,
)
from cosmos_policy.datasets.dataset_utils import (
    calculate_dataset_statistics,
    preprocess_image,
    rescale_data,
    resize_images,
)
from cosmos_policy.utils.utils import duplicate_array

# Set floating point precision to 3 decimal places and disable line wrapping
np.set_printoptions(precision=3, linewidth=np.inf)


class LeRobotPolicyDataset(Dataset):
    def __init__(
        self,
        repo_id: str,
        chunk_size: int = 16,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images=False,
        normalize_actions=True,
        normalize_proprio=True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        use_wrist_images: bool = True,
        use_third_person_images: bool = True,
        use_proprio: bool = True,
        num_duplicates_per_image: int = 4,
        return_value_function_returns: bool = True,
        gamma: float = 0.99,
        image_keys=(
            "observation.images.top",
            "observation.images.down",
            "observation.images.wrist",
        ),
        state_key: str = "observation.state",
        action_key: str = "action",
        cache_dir: str = "",
        video_backend: str = "pyav",
        **_ignored_config_kwargs,  # absorb inherited-only keys (e.g. data_dir, rollout_data_dir) from config merges
    ):
        """
        Initialize a LeRobot Hub dataset for Cosmos Policy training.

        Args:
            repo_id (str): Hugging Face Hub dataset id (e.g. "jokeru/record_p3_orange_1").
            chunk_size (int): Action chunk size.
            final_image_size (int): Target size for resized (square) images.
            t5_text_embeddings_path (str): Path to precomputed T5 text embeddings pickle.
            normalize_images (bool): Whether to normalize images (return float32) or keep uint8.
            normalize_actions (bool): Whether to rescale actions to [-1, 1].
            normalize_proprio (bool): Whether to rescale proprio to [-1, 1].
            use_image_aug / use_stronger_image_aug (bool): Image augmentation toggles.
            use_wrist_images / use_third_person_images / use_proprio (bool): Modality toggles
                (kept for signature parity with the other dataloaders; all True by default).
            num_duplicates_per_image (int): Images per latent frame for the WAN 2.1 tokenizer.
            return_value_function_returns (bool): If True, emit the value latent segment
                (required to reach state_t=11) and Monte-Carlo returns. Demos always have
                value_function_sample_mask=0, so the value loss is masked during BC training.
            gamma (float): Discount factor for value function returns.
            image_keys (tuple): (primary, secondary, wrist) camera feature keys.
            state_key / action_key (str): Proprio and action feature keys.
            cache_dir (str): Where to cache dataset-statistics JSON. Defaults to
                ~/.cache/cosmos_policy/lerobot/<repo_id>.
        """
        self.repo_id = repo_id
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_wrist_images = use_wrist_images
        self.use_third_person_images = use_third_person_images
        self.use_proprio = use_proprio
        self.num_duplicates_per_image = num_duplicates_per_image
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        self.primary_key, self.secondary_key, self.wrist_key = image_keys
        self.state_key = state_key
        self.action_key = action_key

        # Local writable dir for the statistics cache (the HF cache is read-oriented).
        self.data_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "cosmos_policy", "lerobot", repo_id.replace("/", "__")
        )
        os.makedirs(self.data_dir, exist_ok=True)

        # `revision="main"` skips LeRobot's codebase-version git-tag lookup (this Hub
        # dataset is not tagged with a version). `video_backend="pyav"` avoids torchcodec,
        # whose decoder does not survive the fork in multi-worker DataLoaders.
        self.ds = LeRobotDataset(repo_id, revision="main", video_backend=video_backend)

        # Read non-image columns in bulk as materialized numpy arrays (no video decode
        # needed for statistics). `[:]` forces full materialization — indexing the lazy
        # `Column` objects returned by `hf[col]` does NOT behave like numpy masking.
        cols = self.ds.hf_dataset.with_format("numpy")[:]
        actions = cols[action_key]  # (N, action_dim)
        proprio = cols[state_key]  # (N, proprio_dim)
        episode_index = cols["episode_index"]  # (N,)
        global_index = cols["index"]  # (N,)
        task_index = cols["task_index"]  # (N,)

        # Map task_index -> command string (meta.tasks is a DataFrame: task string -> task_index).
        idx_to_task = {int(ti): str(t) for t, ti in zip(self.ds.meta.tasks.index, self.ds.meta.tasks["task_index"])}

        # Group rows by episode into the per-episode structure shared by the other dataloaders.
        # self.data[ep] = dict(actions, proprio, command, num_steps, global_start, returns)
        self.data = {}
        self.unique_commands = set()
        for ep in tqdm(np.unique(episode_index), desc=f"Indexing {repo_id}"):
            mask = episode_index == ep
            command = idx_to_task[int(task_index[mask][0])]
            num_steps = int(mask.sum())
            self.unique_commands.add(command)
            self.data[int(ep)] = dict(
                actions=actions[mask].astype(np.float32),
                proprio=proprio[mask].astype(np.float32),
                command=command,
                num_steps=num_steps,
                global_start=int(global_index[mask].min()),
                returns=(
                    compute_monte_carlo_returns(num_steps, terminal_reward=1.0, gamma=gamma)
                    if return_value_function_returns
                    else None
                ),
            )

        # Flat global-step -> (episode, relative step) mapping.
        self._step_to_episode_map = {}
        self.num_steps = 0
        for ep, ep_data in self.data.items():
            for i in range(ep_data["num_steps"]):
                self._step_to_episode_map[self.num_steps] = (ep, i)
                self.num_steps += 1
        self.epoch_length = self.num_steps

        # Demos only (no rollouts); kept for parity with the RoboCasa sampling layout.
        self.adjusted_demo_count = self.num_steps
        self.adjusted_success_rollout_count = 0

        # Optionally load precomputed T5 text embeddings.
        if t5_text_embeddings_path != "":
            with open(t5_text_embeddings_path, "rb") as f:
                self.t5_text_embeddings = pickle.load(f)

        # Compute (or load cached) statistics, then normalize actions/proprio in place.
        self.dataset_stats = load_or_compute_dataset_statistics(
            data_dir=self.data_dir,
            data=self.data,
            calculate_dataset_statistics_func=calculate_dataset_statistics,
        )
        if self.normalize_actions or self.normalize_proprio:
            if self.normalize_actions:
                self.data = rescale_data(self.data, self.dataset_stats, "actions")
            if self.normalize_proprio:
                self.data = rescale_data(self.data, self.dataset_stats, "proprio")
            self.dataset_stats_post_norm = load_or_compute_post_normalization_statistics(
                data_dir=self.data_dir,
                data=self.data,
                calculate_dataset_statistics_func=calculate_dataset_statistics,
            )

    def __len__(self):
        return self.epoch_length

    def _decode_frame(self, episode_data, relative_step_idx):
        """Decode (primary, secondary, wrist) frames for one step as (H, W, 3) uint8 RGB."""
        sample = self.ds[episode_data["global_start"] + relative_step_idx]

        def to_uint8(img):  # LeRobot returns CHW float in [0, 1]; resize so cameras share a size
            hwc = (img * 255.0).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
            return resize_images(np.expand_dims(hwc, axis=0), self.final_image_size)[0]

        return (
            to_uint8(sample[self.primary_key]),
            to_uint8(sample[self.secondary_key]),
            to_uint8(sample[self.wrist_key]),
        )

    def __getitem__(self, idx):
        """Fetch the 11-segment image sequence + action chunk for a global step index."""
        episode_idx, relative_step_idx = self._step_to_episode_map[idx % self.num_steps]
        episode_data = self.data[episode_idx]

        future_frame_idx = min(relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1)

        # Decode current and future frames (primary=top, secondary=down, wrist).
        cur_primary, cur_secondary, cur_wrist = self._decode_frame(episode_data, relative_step_idx)
        fut_primary, fut_secondary, fut_wrist = self._decode_frame(episode_data, future_frame_idx)

        image_list = []
        seq_idx = 0

        def add(image, duplicate=True):
            nonlocal seq_idx
            arr = duplicate_array(image, total_num_copies=self.num_duplicates_per_image) if duplicate else image
            image_list.append(arr)
            cur = seq_idx
            seq_idx += 1
            return cur

        blank = np.zeros_like(cur_primary)

        # 1) Blank first input frame (needed for the tokenizer; not duplicated).
        add(np.expand_dims(blank, axis=0), duplicate=False)
        # 2) Current proprio (injected later; placeholder blank image here).
        current_proprio_latent_idx = add(blank) if self.use_proprio else -1
        # 3-5) Current wrist, primary, secondary images.
        current_wrist_image_latent_idx = add(cur_wrist) if self.use_wrist_images else -1
        current_image_latent_idx = add(cur_primary) if self.use_third_person_images else -1
        current_image2_latent_idx = add(cur_secondary) if self.use_third_person_images else -1
        # 6) Action chunk (placeholder blank image).
        action_latent_idx = add(blank)
        # 7) Future proprio.
        future_proprio_latent_idx = add(blank) if self.use_proprio else -1
        # 8-10) Future wrist, primary, secondary images.
        future_wrist_image_latent_idx = add(fut_wrist) if self.use_wrist_images else -1
        future_image_latent_idx = add(fut_primary) if self.use_third_person_images else -1
        future_image2_latent_idx = add(fut_secondary) if self.use_third_person_images else -1
        # 11) Value (placeholder blank image).
        value_latent_idx = add(blank) if self.return_value_function_returns else -1

        images = np.concatenate(image_list, axis=0)
        images = preprocess_image(
            images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )

        action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )

        proprio = episode_data["proprio"][relative_step_idx]
        future_proprio = episode_data["proprio"][future_frame_idx]
        value_function_return = (
            episode_data["returns"][future_frame_idx] if self.return_value_function_returns else float("-100")
        )

        return {
            "video": images,
            "command": episode_data["command"],
            "actions": action_chunk,
            "t5_text_embeddings": torch.squeeze(self.t5_text_embeddings[episode_data["command"]]),
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": proprio if self.use_proprio else np.zeros_like(episode_data["proprio"][relative_step_idx]),
            "future_proprio": (
                future_proprio if self.use_proprio else np.zeros_like(episode_data["proprio"][future_frame_idx])
            ),
            "__key__": idx,
            # Demos only: no rollouts, value loss masked.
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": value_latent_idx,
            "current_proprio_latent_idx": current_proprio_latent_idx,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx,
            "current_image_latent_idx": current_image_latent_idx,
            "current_image2_latent_idx": current_image2_latent_idx,
            "future_proprio_latent_idx": future_proprio_latent_idx,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx,
            "future_image_latent_idx": future_image_latent_idx,
            "future_image2_latent_idx": future_image2_latent_idx,
            "value_function_return": value_function_return,
        }


if __name__ == "__main__":
    dataset = LeRobotPolicyDataset(repo_id="jokeru/record_p3_orange_1")

    # Inject dummy T5 embeddings so the smoke test runs without loading the T5 model.
    dataset.t5_text_embeddings = {cmd: torch.zeros(1, 512, 1024) for cmd in dataset.unique_commands}

    print(f"\nNum episodes: {len(dataset.data)}, num steps: {dataset.num_steps}")
    print(f"Unique commands: {dataset.unique_commands}")

    sample = dataset[100]
    print(f"\nVideo shape, dtype: {sample['video'].shape, sample['video'].dtype}")
    print(f"Actions shape, dtype: {sample['actions'].shape, sample['actions'].dtype}")
    print("Latent indices:")
    for k, v in sample.items():
        if k.endswith("_latent_idx"):
            print(f"  {k}: {v}")
