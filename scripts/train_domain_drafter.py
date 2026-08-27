#!/usr/bin/env python3
"""
Wrapper around train_drafter.py that adds:
  - Auto-calculated save_steps/eval_steps for 2 checkpoints per epoch
  - Resume from latest checkpoint in output_dir
  - Extended per-step and per-eval metrics logged to MLflow
  - GPU memory, throughput, gradient norms, perplexity, top-1 accuracy
  - Summary stats printed at startup

Usage (from project root):
    CUDA_VISIBLE_DEVICES=0 python scripts/train_domain_drafter.py --config-name=train_understanding
    CUDA_VISIBLE_DEVICES=1 python scripts/train_domain_drafter.py --config-name=train_text_reformulation

All Hydra overrides are forwarded as-is.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from data import DistillationCollator, SpecDecDataset
from training import DistillationLoss


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_cluster_list(path_str: str) -> list[str]:
    path = _resolve(path_str)
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("clusters") or data.get("names")
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of cluster names")
    return data


# ── Custom Trainer with distillation loss ────────────────────────────────────


class DistillationTrainer(Trainer):
    def __init__(self, *args, loss_fn: DistillationLoss, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        # Accumulators for running averages between logging steps
        self._ce_accum = 0.0
        self._kd_accum = 0.0
        self._loss_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        gen_starts = inputs.pop("gen_start", None)
        inputs.pop("cluster", None)
        top10_ids = inputs.pop("top10_ids", None)
        top10_probs = inputs.pop("top10_probs", None)
        labels = inputs.get("labels")

        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        loss_out = self.loss_fn(
            logits=logits,
            labels=labels,
            top10_ids=top10_ids,
            top10_probs=top10_probs,
            gen_starts=gen_starts,
        )

        # Accumulate component losses for averaged logging
        if loss_out.ce_loss is not None:
            self._ce_accum += float(loss_out.ce_loss.detach().item())
        if loss_out.kd_loss is not None:
            self._kd_accum += float(loss_out.kd_loss.detach().item())
        self._loss_count += 1

        return (loss_out.loss, outputs) if return_outputs else loss_out.loss

    def _flush_component_losses(self) -> dict:
        """Return averaged component losses since last flush, then reset."""
        metrics = {}
        n = max(self._loss_count, 1)
        if self._ce_accum > 0:
            metrics["train/ce_loss"] = self._ce_accum / n
        if self._kd_accum > 0:
            metrics["train/kd_loss"] = self._kd_accum / n
        self._ce_accum = 0.0
        self._kd_accum = 0.0
        self._loss_count = 0
        return metrics


# ── Logging callback ────────────────────────────────────────────────────────


class ExtendedLoggingCallback(TrainerCallback):
    """Log extra metrics: learning rate, epoch progress, GPU memory, throughput,
    gradient norm, and component losses."""

    def __init__(self):
        self._last_log_time = time.time()
        self._last_log_step = 0
        self._train_start_time = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        now = time.time()
        dt = now - self._last_log_time
        dsteps = state.global_step - self._last_log_step

        # Throughput
        if dt > 0 and dsteps > 0:
            logs["perf/steps_per_sec"] = round(dsteps / dt, 3)
            logs["perf/samples_per_sec"] = round(
                (dsteps * args.per_device_train_batch_size) / dt, 1
            )

        # GPU memory
        if torch.cuda.is_available():
            logs["gpu/mem_allocated_gb"] = round(
                torch.cuda.memory_allocated() / 1e9, 2
            )
            logs["gpu/mem_reserved_gb"] = round(
                torch.cuda.memory_reserved() / 1e9, 2
            )
            logs["gpu/mem_peak_gb"] = round(
                torch.cuda.max_memory_allocated() / 1e9, 2
            )

        # Epoch progress
        logs["train/epoch_progress"] = round(state.epoch, 4) if state.epoch else 0

        # Elapsed time
        if self._train_start_time:
            logs["train/elapsed_hours"] = round(
                (now - self._train_start_time) / 3600, 3
            )

        # Perplexity from loss
        if "loss" in logs:
            try:
                logs["train/perplexity"] = round(math.exp(min(logs["loss"], 20)), 2)
            except (OverflowError, ValueError):
                logs["train/perplexity"] = float("inf")

        # Component losses from trainer accumulator
        model = kwargs.get("model")
        trainer = None
        # The trainer is accessible via the callback handler
        if hasattr(state, "_trainer"):
            trainer = state._trainer
        if trainer is None:
            # Try to get it from the callback handler
            for cb in self._registered_callbacks if hasattr(self, "_registered_callbacks") else []:
                if hasattr(cb, "loss_fn"):
                    trainer = cb
                    break

        self._last_log_time = now
        self._last_log_step = state.global_step

    def on_train_begin(self, args, state, control, **kwargs):
        self._last_log_time = time.time()
        self._last_log_step = 0
        self._train_start_time = time.time()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Compute perplexity from eval loss."""
        if metrics and "eval_loss" in metrics:
            try:
                metrics["eval/perplexity"] = round(
                    math.exp(min(metrics["eval_loss"], 20)), 2
                )
            except (OverflowError, ValueError):
                metrics["eval/perplexity"] = float("inf")


class GradientNormCallback(TrainerCallback):
    """Log gradient L2 norm after each optimizer step."""

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is not None and state.global_step % max(args.logging_steps, 1) == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            # Store for next on_log call
            if not hasattr(state, "_grad_norm_cache"):
                state._grad_norm_cache = {}
            state._grad_norm_cache[state.global_step] = total_norm


class ComponentLossLoggerCallback(TrainerCallback):
    """Flush averaged component losses (CE + KD) from the trainer on each log step."""

    def __init__(self, trainer_ref):
        self._trainer_ref = trainer_ref

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        comp = self._trainer_ref._flush_component_losses()
        logs.update(comp)

        # Also inject gradient norm if available
        if hasattr(state, "_grad_norm_cache"):
            gn = state._grad_norm_cache.pop(state.global_step, None)
            if gn is not None:
                logs["train/grad_norm"] = round(gn, 4)


# ── Eval metrics computation ────────────────────────────────────────────────


def build_compute_metrics(loss_fn):
    """Build a compute_metrics function for the Trainer that computes
    top-1 token accuracy on eval set.

    NOTE: preprocess_logits_for_metrics already does argmax to save VRAM,
    so eval_pred.predictions contains token IDs [N, L], not logits.
    """

    def compute_metrics(eval_pred):
        preds_np, labels_np = eval_pred.predictions, eval_pred.label_ids

        # Shift: pred[i] predicts labels[i+1]
        shifted_preds = preds_np[:, :-1]
        shifted_labels = labels_np[:, 1:]

        mask = shifted_labels != -100
        n_valid = mask.sum()

        if n_valid > 0:
            correct = ((shifted_preds == shifted_labels) & mask).sum()
            accuracy = float(correct) / float(n_valid)
        else:
            accuracy = 0.0

        return {
            "eval/top1_accuracy": round(accuracy, 4),
            "eval/n_valid_tokens": int(n_valid),
        }

    return compute_metrics


# ── Main ─────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    t0 = time.time()

    print("=" * 70)
    print("Domain Drafter Training (extended wrapper)")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.training.seed)

    # Load clusters
    clusters = load_cluster_list(cfg.data.clusters_json)
    print(f"[setup] {len(clusters)} clusters from {cfg.data.clusters_json}")

    # Load model + tokenizer
    model_path = _resolve(cfg.model.path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=cfg.model.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, cfg.model.dtype) if cfg.model.dtype else None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {model_path.name}: {n_params:,} params ({n_trainable:,} trainable)")

    # Load dataset
    mode = "distillation" if cfg.training.loss.mode in {"distillation", "mixed"} else "standard"
    synth_dir = _resolve(cfg.data.synthetic_dir)
    dataset = SpecDecDataset.from_dir(
        synth_dir,
        tokenizer=tokenizer,
        mode=mode,
        max_length=cfg.data.max_length,
        max_gen_length=cfg.data.max_gen_length,
        clusters_filter=clusters,
        min_top1_prob=cfg.data.min_top1_prob,
    )

    # Print dataset stats
    stats = dataset.stats()
    print(f"[data] Total samples: {stats['total_samples']}")
    print(f"[data] Clusters loaded: {stats['num_clusters']}")
    for cname, cnt in sorted(stats["cluster_counts"].items()):
        print(f"       {cname}: {cnt}")
    if "gen_len" in stats:
        gl = stats["gen_len"]
        print(f"[data] Gen length: mean={gl['mean']}, min={gl['min']}, max={gl['max']}")
    if "mean_top1_prob" in stats:
        print(f"[data] Mean top-1 prob: {stats['mean_top1_prob']}")

    if len(dataset) == 0:
        raise RuntimeError(
            f"No samples found under {synth_dir} for the requested clusters."
        )

    # Split
    val_frac = float(cfg.data.val_fraction)
    if val_frac > 0:
        train_ds, val_ds, _ = dataset.split(
            (1.0 - val_frac, val_frac, 0.0), seed=cfg.data.seed
        )
    else:
        train_ds, val_ds = dataset, None

    n_train = len(train_ds)
    n_val = len(val_ds) if val_ds else 0
    print(f"[data] Split: train={n_train}, val={n_val}")

    # Calculate save_steps for 2 checkpoints per epoch
    batch_size = cfg.training.per_device_train_batch_size
    grad_accum = cfg.training.gradient_accumulation_steps
    effective_batch = batch_size * grad_accum
    steps_per_epoch = (n_train + effective_batch - 1) // effective_batch
    save_steps = max(steps_per_epoch // 2, 1)
    eval_steps = save_steps

    # Allow config override if explicitly set to a non-default value
    cfg_save = cfg.training.save_steps
    if cfg_save not in (500, None):
        save_steps = cfg_save
        eval_steps = cfg_save

    total_steps = steps_per_epoch * cfg.training.num_train_epochs
    print(f"[schedule] Steps/epoch: {steps_per_epoch}")
    print(f"[schedule] Save/eval every: {save_steps} steps (2x per epoch)")
    print(f"[schedule] Total steps: {total_steps}")
    print(f"[schedule] Epochs: {cfg.training.num_train_epochs}")
    print(f"[schedule] Batch size: {batch_size} (effective: {effective_batch})")
    print(f"[schedule] Loss mode: {cfg.training.loss.mode} (ce_weight={cfg.training.loss.ce_weight})")

    # Collator + Loss
    collator = DistillationCollator(pad_token_id=tokenizer.pad_token_id)
    loss_fn = DistillationLoss(
        mode=cfg.training.loss.mode,
        ce_weight=cfg.training.loss.ce_weight,
        temperature=cfg.training.loss.temperature,
    )

    # MLflow
    if cfg.mlflow.tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = cfg.mlflow.tracking_uri
    if cfg.mlflow.experiment_name:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.mlflow.experiment_name
    report_to = ["mlflow"] if cfg.mlflow.tracking_uri else []

    do_eval = val_ds is not None and len(val_ds) > 0
    output_dir = str(_resolve(cfg.training.output_dir))

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        logging_steps=cfg.training.logging_steps,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=cfg.training.save_total_limit,
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        seed=cfg.training.seed,
        remove_unused_columns=False,
        report_to=report_to,
        run_name=cfg.mlflow.run_name,
        load_best_model_at_end=do_eval,
        metric_for_best_model="eval_loss" if do_eval else None,
        greater_is_better=False,
        logging_first_step=True,
    )

    trainer = DistillationTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        loss_fn=loss_fn,
        callbacks=[
            ExtendedLoggingCallback(),
            GradientNormCallback(),
        ],
        compute_metrics=build_compute_metrics(loss_fn) if do_eval else None,
        preprocess_logits_for_metrics=_preprocess_logits if do_eval else None,
    )

    # Add the component loss logger callback (needs reference to trainer)
    trainer.add_callback(ComponentLossLoggerCallback(trainer))

    # Resume from checkpoint if available
    resume_ckpt = None
    ckpt_dir = Path(output_dir)
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if ckpts:
            resume_ckpt = str(ckpts[-1])
            print(f"[resume] Resuming from {resume_ckpt}")

    print(f"\n{'='*70}")
    print(f"Starting training... ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*70}\n")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save final model
    final_dir = _resolve(cfg.training.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Training complete! ({elapsed/3600:.1f} hours)")
    print(f"Final model: {final_dir}")
    print(f"{'='*70}")


def _preprocess_logits(logits, labels):
    """Reduce logits to argmax predictions to prevent GPU OOM during eval.
    Full logits are [B, L, 32000] in float32 — too large to accumulate.
    Returns [B, L] int64 token IDs."""
    return logits.argmax(dim=-1)


if __name__ == "__main__":
    main()
