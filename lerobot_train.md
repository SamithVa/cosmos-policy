# Train Cosmos Policy on the LeRobot dataset

Overrides must come AFTER a `--` separator and be bare dotlist items
(`experiment=...`, not `--experiment=...`). Lines are joined with trailing `\`.

```bash
export BASE_DATASETS_DIR=/home/data/wanshan/data

CUDA_VISIBLE_DEVICES=0,2,3,6 torchrun --nproc_per_node=4 --master_port=12341 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_lerobot" \
  trainer.grad_accum_iter=4
```

## Validate the config first (no training, fast)

```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=12341 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py --dryrun -- \
  experiment="cosmos_predict2_2b_480p_lerobot"
```

## Single-GPU smoke test (a few steps, finite loss)

```bash

export IMAGINAIRE_OUTPUT_ROOT=/home/data/wanshan/imaginaire4-output

CUDA_VISIBLE_DEVICES=0,6 torchrun --nproc_per_node=2 --master_port=12341 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_lerobot" \
  trainer.max_iter=100000
```

export BASE_DATASETS_DIR=/home/data/wanshan/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=4,5,6 torchrun --nproc_per_node=3 --master_port=12341 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_lerobot" \
  trainer.grad_accum_iter=8
```
