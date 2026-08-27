# Training Operations — Safe Launch Guide

This document describes how to launch training so it does NOT kill the VibeIDE bot process.

## The Problem

The bot (vibeide.service) and Claude Agent SDK's Bash tool run in the same systemd cgroup. When `python scripts/train_domain_drafter.py` is launched directly via Bash, its memory consumption is attributed to the bot's cgroup. If the training script OOMs, the Linux OOM killer may kill the bot's Node.js process too — causing a full restart and loss of context.

## Solution: Launch training via `systemd-run --user --scope`

This places the training process in a **separate cgroup**, isolated from the bot. If training OOMs, only the training process dies — the bot stays alive and sees the error.

### Single domain

```bash
# Understanding (GPU 0)
systemd-run --user --scope -p MemoryMax=58G \
  bash -c 'cd /home/rudenko/multisd/Domain-Aware-SD && \
    source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate domain_sd && \
    export MLFLOW_ALLOW_FILE_STORE=true && \
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
    CUDA_VISIBLE_DEVICES=0 python scripts/train_domain_drafter.py \
      --config-name=train_understanding'

# Text Reformulation (GPU 1)
systemd-run --user --scope -p MemoryMax=58G \
  bash -c 'cd /home/rudenko/multisd/Domain-Aware-SD && \
    source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate domain_sd && \
    export MLFLOW_ALLOW_FILE_STORE=true && \
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
    CUDA_VISIBLE_DEVICES=1 python scripts/train_domain_drafter.py \
      --config-name=train_text_reformulation'
```

### Both domains in parallel

```bash
# Launch understanding on GPU 0 in a separate cgroup
systemd-run --user --scope --unit=train-understanding -p MemoryMax=30G \
  bash -c 'cd /home/rudenko/multisd/Domain-Aware-SD && \
    source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate domain_sd && \
    export MLFLOW_ALLOW_FILE_STORE=true && \
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
    CUDA_VISIBLE_DEVICES=0 python scripts/train_domain_drafter.py \
      --config-name=train_understanding' \
  > logs/train_understanding.log 2>&1 &

# Launch text_reformulation on GPU 1 in a separate cgroup
systemd-run --user --scope --unit=train-text-reformulation -p MemoryMax=30G \
  bash -c 'cd /home/rudenko/multisd/Domain-Aware-SD && \
    source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate domain_sd && \
    export MLFLOW_ALLOW_FILE_STORE=true && \
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
    CUDA_VISIBLE_DEVICES=1 python scripts/train_domain_drafter.py \
      --config-name=train_text_reformulation' \
  > logs/train_text_reformulation.log 2>&1 &

# Wait for both
wait
```

### With nohup (survives bot restart)

If you want training to survive even a bot restart:

```bash
nohup systemd-run --user --scope --unit=train-understanding -p MemoryMax=30G \
  bash -c 'cd /home/rudenko/multisd/Domain-Aware-SD && \
    source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate domain_sd && \
    export MLFLOW_ALLOW_FILE_STORE=true && \
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
    CUDA_VISIBLE_DEVICES=0 python scripts/train_domain_drafter.py \
      --config-name=train_understanding' \
  > logs/train_understanding.log 2>&1 &
```

## Monitoring

```bash
# Check if training is running
pgrep -af train_domain_drafter

# Check GPU usage
nvidia-smi

# Check memory of training cgroup
systemctl --user status run-*.scope

# Tail training logs
tail -f /home/rudenko/multisd/Domain-Aware-SD/logs/train_understanding.log
tail -f /home/rudenko/multisd/Domain-Aware-SD/logs/train_text_reformulation.log
```

## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `MemoryMax` | 30G per domain | Set based on available RAM. Total system has ~64G. Leave ~4G for bot + OS. |
| `dataloader_num_workers` | 0 | Keep at 0 for in-memory datasets — fork workers duplicate data in RAM. |
| `per_device_train_batch_size` | 32 | Reduce to 16 or 8 if GPU OOM occurs. |

## What NOT to do

- **Do NOT** run `python scripts/train_domain_drafter.py` directly from the Claude Bash tool — it shares the bot's cgroup and OOM will kill the bot.
- **Do NOT** run `bash scripts/run_domain_training.sh` directly — same problem, it launches training in the bot's cgroup.
- **Do NOT** set `dataloader_num_workers > 0` — each worker forks the process and duplicates the dataset in RAM.
- **Do NOT** run both domains in parallel without `-p MemoryMax=30G` — together they can consume 50G+ and OOM the system.

---

## Experiment Results

### Understanding domain (21 clusters)

**Config**: `configs/train_understanding.yaml`
**Model**: Lite-Mistral-150M-v2-Instruct (156M params)
**Data**: 395,280 samples (21 clusters × Flan tasks), split 95/5 train/val
**Loss**: mixed (0.5 × CE + 0.5 × KL-divergence against target top-10)
**Hardware**: 1× RTX 3090 (24GB), CUDA_VISIBLE_DEVICES=0
**Batch size**: 32, bf16, cosine LR schedule, lr=5e-5, warmup 3%
**Training time**: ~4.3 hours (stopped early at epoch 4.5 — plateau)

#### Eval metrics (on 5% held-out val set, 19,762 samples):

| Epoch | eval_loss | top1_accuracy | Status |
|-------|-----------|---------------|--------|
| 0.5   | 1.840     | 62.0%         |        |
| 1.5   | 1.282     | 64.8%         |        |
| 2.0   | 1.282     | 64.8%         | plateau start |
| 2.5   | 1.232     | 64.8%         |        |
| 3.0   | 1.213     | 64.9%         |        |
| 3.5   | 1.202     | 65.0%         |        |
| 4.0   | 1.196     | 65.0%         |        |
| **4.5** | **1.193** | **65.05%**  | **best, stopped** |

**Best checkpoint**: `checkpoint-52803` (epoch 4.5)
**Saved to**: `/media/public/rudenko/projects/Domain-Aware-SD/outputs/drafter_understanding/checkpoint-52803`

#### Key observations:
- Model reached plateau around epoch 2.0 — eval_loss improvement slowed from 0.56/epoch (0→2) to 0.02/epoch (2→4.5)
- Top-1 accuracy stabilized at ~65%, suggesting the drafter captures about 2/3 of the target's token-level decisions
- Training loss continued to decrease (1.43 → 1.19) while eval loss plateaued — mild overfitting starting

### Text Reformulation domain (11 clusters)

**Config**: `configs/train_text_reformulation.yaml`
**Planned**: 5 epochs (based on Understanding plateau at ~4.5 epochs)
**Output**: `/media/public/rudenko/projects/Domain-Aware-SD/outputs/drafter_text_reformulation/`

#### Reproducibility

```bash
# Understanding (from project root)
CUDA_VISIBLE_DEVICES=0 MLFLOW_ALLOW_FILE_STORE=true \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/rudenko/miniconda3/envs/domain_sd/bin/python \
  scripts/train_domain_drafter.py --config-name=train_understanding

# Text Reformulation (from project root)
CUDA_VISIBLE_DEVICES=0 MLFLOW_ALLOW_FILE_STORE=true \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/rudenko/miniconda3/envs/domain_sd/bin/python \
  scripts/train_domain_drafter.py --config-name=train_text_reformulation
```
