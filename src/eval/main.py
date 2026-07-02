"""
Hydra entry point for the evaluation pipeline.

Usage (from project root):
    python src/eval/main.py
    python src/eval/main.py target_source=model
    python src/eval/main.py draft_models=[tiny-mixtral,another-drafter]
    python src/eval/main.py 'datasets=[data/synthetic/v1/aeslc_10templates.jsonl]'
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import hydra
import mlflow
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

# Allow running as `python src/eval/main.py` from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import SpecDecDataset  # noqa: E402
from src.data.utils import walk_data_files  # noqa: E402
from src.eval import mlflow_logger  # noqa: E402
from src.eval.draft_runner import DraftRunner  # noqa: E402
from src.eval.evaluator import evaluate_dataset  # noqa: E402
from src.eval.model_loader import resolve_model_dir  # noqa: E402
from src.eval.output_writer import (  # noqa: E402
    EvalMetadata,
    mode_label,
    output_filename,
    write_eval_json,
)
from src.eval.target_provider import (  # noqa: E402
    DatasetTargetProvider,
    ModelTargetProvider,
)

load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _auto_run_name(draft_models: List[str], dataset_path: str) -> str:
    drafts = "+".join(draft_models) if len(draft_models) <= 3 else f"multi-{len(draft_models)}"
    stem = Path(dataset_path).stem
    return f"{drafts}__{stem}"


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print("Effective config:")
    print(OmegaConf.to_yaml(cfg))

    # ── Resolve target model + tokenizer (target tokenizer drives all dataset
    # tokenization; draft tokenizer is assumed to share IDs 0..V_draft-1 with it).
    target_dir = _resolve_path(cfg.target.path)
    if not target_dir.exists():
        sys.exit(f"Target model directory not found: {target_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(target_dir), trust_remote_code=cfg.target.trust_remote_code
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # ── Resolve all draft model dirs (local → S3) ────────────────────────────
    draft_dirs: List[Path] = []
    for name in cfg.draft_models:
        d = resolve_model_dir(
            name, bucket=cfg.s3.bucket, s3_prefix=cfg.s3.models_prefix
        )
        draft_dirs.append(d)

    # ── Load draft runners (all at once; assumes small drafts) ──────────────
    draft_runners: List[DraftRunner] = []
    print(f"\nLoading {len(draft_dirs)} draft model(s)...")
    for d in draft_dirs:
        print(f"  - {d}")
        runner = DraftRunner(
            model_dir=d,
            device=cfg.device,
            dtype=cfg.dtype,
            topk_K=cfg.topk_K,
            trust_remote_code=False,
        )
        draft_runners.append(runner)
    draft_vocab_size = draft_runners[0].vocab_size

    # ── Target provider ──────────────────────────────────────────────────────
    target_source = cfg.target_source
    if target_source == "dataset":
        target_provider = DatasetTargetProvider()
    elif target_source == "model":
        print(f"\nLoading target model: {target_dir}")
        target_provider = ModelTargetProvider(
            model_dir=target_dir,
            device=cfg.device,
            dtype=cfg.dtype,
            trust_remote_code=cfg.target.trust_remote_code,
            topk_K=cfg.topk_K,
        )
    else:
        sys.exit(f"Unknown target_source={target_source!r} (use 'dataset' or 'model')")

    # ── Per-dataset eval loop ────────────────────────────────────────────────
    output_dir = _resolve_path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Expand any directory entries in cfg.datasets into all *.jsonl / *.npz
    # files inside them (recursively). File entries pass through unchanged.
    expanded_datasets: List[Path] = []
    for ds_path_str in cfg.datasets:
        p = _resolve_path(ds_path_str)
        if not p.exists():
            print(f"  [skip] path not found: {p}")
            continue
        if p.is_dir():
            files = walk_data_files(p)
            if not files:
                print(f"  [skip] no *.jsonl / *.npz under directory: {p}")
                continue
            print(f"  [expand] {p} → {len(files)} file(s)")
            expanded_datasets.extend(files)
        else:
            expanded_datasets.append(p)

    if not expanded_datasets:
        sys.exit("No datasets to evaluate.")

    for ds_path in expanded_datasets:
        print(f"\n── Evaluating on {ds_path} ──")

        dataset = SpecDecDataset(
            ds_path,
            tokenizer=tokenizer,
            mode="distillation",
            # Cap trunk length to n_positions (max_gen_length); allow plenty of
            # room for the prompt by setting max_length conservatively.
            max_length=2048,
            max_gen_length=cfg.n_positions,
        )

        run_name = cfg.mlflow.run_name or _auto_run_name(list(cfg.draft_models), str(ds_path))
        mlflow_logger.start_run(
            tracking_uri=cfg.mlflow.tracking_uri,
            experiment_name=cfg.mlflow.experiment_name,
            run_name=run_name,
        )
        try:
            mlflow_logger.log_params_from_config(cfg_dict)
            mlflow.log_param("dataset_basename", Path(ds_path).stem)

            top_k_from_dataset = 10 if target_source == "dataset" else None
            mode_str = mode_label(target_source, top_k_from_dataset)

            result = evaluate_dataset(
                dataset=dataset,
                draft_runners=draft_runners,
                target_provider=target_provider,
                n_positions=cfg.n_positions,
                batch_size=cfg.batch_size,
                pad_token_id=pad_id,
                metrics=list(cfg.metrics),
                aggregations=list(cfg.aggregations),
                mode_label_str=mode_str,
                draft_vocab_size=draft_vocab_size,
                max_samples=cfg.max_samples_per_file,
            )

            print(
                f"  processed {result.n_samples_total} samples; "
                f"skipped positions total: {sum(result.n_skipped_per_position)}; "
                f"special-target positions total: {sum(result.n_special_target_per_position)}; "
                f"samples with OOB in trunk: {result.n_samples_with_oob_in_trunk}"
            )

            aggs_used = ["single"] if len(draft_runners) == 1 else list(cfg.aggregations)
            metadata = EvalMetadata(
                draft_models=list(cfg.draft_models),
                target_model=cfg.target.path,
                target_source=target_source,
                dataset=str(ds_path),
                n_positions=cfg.n_positions,
                n_samples_total=result.n_samples_total,
                n_samples_per_position=result.n_samples_per_position,
                n_skipped_per_position=result.n_skipped_per_position,
                n_special_target_per_position=result.n_special_target_per_position,
                n_samples_with_oob_in_trunk=result.n_samples_with_oob_in_trunk,
                top_k_from_dataset=top_k_from_dataset,
                target_renormalized=True,
                metrics=list(cfg.metrics),
                aggregations=aggs_used,
                config_snapshot=cfg_dict,
            )
            out_path = write_eval_json(
                output_dir,
                list(cfg.draft_models),
                str(ds_path),
                metadata,
                result.results,
            )
            print(f"  wrote {out_path}")

            mlflow_logger.log_results(result.results, result.n_samples_per_position)
            mlflow_logger.log_artifact(out_path)
        finally:
            mlflow.end_run()


if __name__ == "__main__":
    # MLflow file-store opt-in (matches generate_synthetic_data.py convention).
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    main()
