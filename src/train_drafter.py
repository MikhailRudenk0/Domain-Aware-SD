#!/usr/bin/env python3
"""
Fine-tune a drafter model on synthetic data produced by the target.

Trains ONE drafter per invocation on the union of clusters listed in a JSON
file (see configs/clusters/all.json for an example). Supports three losses:
  ce            — next-token cross-entropy on the target's trunk samples
  distillation  — soft cross-entropy against the target's top-10 distribution
  mixed         — weighted sum of both

Usage:
    python src/train_drafter.py
    python src/train_drafter.py data.clusters_json=configs/clusters/summarization.json
    python src/train_drafter.py training.loss.mode=mixed training.loss.ce_weight=0.3
    python src/train_drafter.py model.path=Lite-Mistral-150M-v2-Instruct
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)

from data import DistillationCollator, SpecDecDataset  # noqa: E402
from training import DistillationLoss  # noqa: E402


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_cluster_list(path_str: str) -> List[str]:
    path = _resolve(path_str)
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("clusters") or data.get("names")
    if not isinstance(data, list) or not all(isinstance(c, str) for c in data):
        raise ValueError(
            f"{path} must contain a JSON list of cluster names "
            f"(or a dict with a 'clusters' / 'names' list field)."
        )
    return data


class DistillationTrainer(Trainer):
    """HuggingFace Trainer with a pluggable CE/KD/mixed loss."""

    def __init__(self, *args, loss_fn: DistillationLoss, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

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

        if loss_out.ce_loss is not None or loss_out.kd_loss is not None:
            self._log_component_losses(loss_out)

        return (loss_out.loss, outputs) if return_outputs else loss_out.loss

    def _log_component_losses(self, loss_out) -> None:
        # Best-effort side-log of ce/kd components; the primary loss is logged
        # by Trainer as usual.
        try:
            if self.state.global_step % max(self.args.logging_steps, 1) != 0:
                return
            comp = {}
            if loss_out.ce_loss is not None:
                comp["ce_loss"] = float(loss_out.ce_loss.detach().item())
            if loss_out.kd_loss is not None:
                comp["kd_loss"] = float(loss_out.kd_loss.detach().item())
            if comp:
                self.log(comp)
        except Exception:
            pass


def build_model_and_tokenizer(cfg: DictConfig):
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
    return model, tokenizer


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    print("[train_drafter] Config:")
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.training.seed)

    clusters = load_cluster_list(cfg.data.clusters_json)
    print(f"[train_drafter] {len(clusters)} clusters requested from {cfg.data.clusters_json}")

    model, tokenizer = build_model_and_tokenizer(cfg)

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
    print(
        f"[train_drafter] Loaded {len(dataset)} samples "
        f"across {len(dataset.cluster_names())} clusters"
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No samples found under {synth_dir} for the requested clusters. "
            "Check that generate_synthetic_data.py has run and produced matching files."
        )

    val_frac = float(cfg.data.val_fraction)
    if val_frac > 0:
        train_ds, val_ds, _ = dataset.split((1.0 - val_frac, val_frac, 0.0), seed=cfg.data.seed)
    else:
        train_ds, val_ds = dataset, None
    print(
        f"[train_drafter] Split: train={len(train_ds)} "
        f"val={len(val_ds) if val_ds is not None else 0}"
    )

    collator = DistillationCollator(pad_token_id=tokenizer.pad_token_id)
    loss_fn = DistillationLoss(
        mode=cfg.training.loss.mode,
        ce_weight=cfg.training.loss.ce_weight,
        temperature=cfg.training.loss.temperature,
    )

    if cfg.mlflow.tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = cfg.mlflow.tracking_uri
    if cfg.mlflow.experiment_name:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.mlflow.experiment_name
    report_to = ["mlflow"] if cfg.mlflow.tracking_uri else []

    do_eval = val_ds is not None and len(val_ds) > 0

    args = TrainingArguments(
        output_dir=str(_resolve(cfg.training.output_dir)),
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        logging_steps=cfg.training.logging_steps,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=cfg.training.eval_steps,
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
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
    )

    trainer = DistillationTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        loss_fn=loss_fn,
    )

    trainer.train()

    final_dir = _resolve(cfg.training.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[train_drafter] Saved final drafter to {final_dir}")


if __name__ == "__main__":
    main()
