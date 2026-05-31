# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Precomputes T5 text embeddings for a LeRobot Hub dataset's task descriptions and saves
them to disk for faster training.

Usage:
    uv run -m cosmos_policy.datasets.save_lerobot_t5_text_embeddings \
        --repo_id jokeru/record_p3_orange_1 [--out_dir OUT_DIR]
"""

import argparse

import cosmos_policy._src.predict2.inference.get_t5_emb as t5
from cosmos_policy.datasets.lerobot_dataset import LeRobotPolicyDataset
from cosmos_policy.datasets.t5_embedding_utils import (
    generate_t5_embeddings,
    save_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute T5 text embeddings for a LeRobot dataset")
    parser.add_argument("--repo_id", type=str, default="jokeru/record_p3_orange_1", help="HF Hub dataset id")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Directory to write t5_embeddings.pkl (defaults to the dataset's local cache dir)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Reuse the T5 encoder already in the HF default cache (local_files_only avoids
    # contacting the Hub and re-downloading a newer safetensors revision).
    print("Loading T5-11B encoder from local cache (~45GB, may take a minute)...", flush=True)
    t5.cosmos_encoder = t5.CosmosT5TextEncoder(local_files_only=True)
    print("T5 encoder loaded.", flush=True)

    print("Loading data...")
    dataset = LeRobotPolicyDataset(repo_id=args.repo_id)

    out_dir = args.out_dir or dataset.data_dir
    t5_text_embeddings = generate_t5_embeddings(dataset.unique_commands)
    save_embeddings(t5_text_embeddings, out_dir)


if __name__ == "__main__":
    main()
