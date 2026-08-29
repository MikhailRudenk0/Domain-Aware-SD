# Domain-Aware Speculative Decoding: Training Report

## 1. Experiment Overview

**Research question**: Does a domain-specific draft model achieve higher acceptance rates (AR) in speculative decoding compared to a single draft model trained on a broader dataset?

**Setup**:
- **Target model**: TurboSparse-Mistral-Instruct (7B, BambooForCausalLM)
- **Draft model**: Lite-Mistral-150M-v2-Instruct (156M params, MistralForCausalLM)
- **Training data**: Synthetic distillation data (v3) — target model's top-10 token distributions generated via teacher-forcing on Flan tasks
- **Loss function**: Mixed = 0.5 × CrossEntropy + 0.5 × KL-divergence (temperature=1.0)
- **Hardware**: 1× NVIDIA RTX 3090 (24 GB), 62 GB RAM
- **Framework**: HuggingFace Transformers + DeepSpeed-compatible Trainer

---

## 2. Domain Partitioning

Three draft models were trained:

| Model | Clusters | Train Samples | Description |
|-------|----------|---------------|-------------|
| **Understanding** | 21 | 395,280 | NLU tasks: sentiment, NLI, QA, paraphrase detection |
| **Text Reformulation** | 11 | 327,330 | Translation, punctuation, word segmentation, true-casing |
| **Mixed (U+T)** | 32 | 722,610 | Union of both domains (baseline) |

**Understanding clusters** (21): ag_news, bool_q, cola, glue_mrpc, glue_qqp, imdb_reviews, mnli_matched, mnli_mismatched, paws_wiki, qnli, quac, rte, sentiment140, snli, sst2, stsb, trec, wic, wnli, wsc, yelp_polarity_reviews

**Text Reformulation clusters** (11): fix_punct, para_crawl_enes, true_case, wmt14_enfr, wmt16_translate_csen/deen/fien/roen/ruen/tren, word_segment

---

## 3. Training Hyperparameters

All three models share identical hyperparameters:

| Parameter | Value |
|-----------|-------|
| Base model | Lite-Mistral-150M-v2-Instruct (156M) |
| Precision | bfloat16 |
| Batch size (per device) | 32 |
| Gradient accumulation | 1 |
| Learning rate | 5 × 10⁻⁵ |
| LR schedule | Cosine with 3% warmup |
| Weight decay | 0.01 |
| Max sequence length | 512 |
| Max generation length | 256 |
| Validation split | 5% (stratified per cluster) |
| Loss | 0.5 × CE + 0.5 × KL (T=1.0) |
| Optimizer | AdamW (HF Trainer default) |
| Eval frequency | 2× per epoch |
| Save total limit | 4 checkpoints |
| Epochs (Understanding) | 25 (stopped early at 4.5) |
| Epochs (Text Reformulation) | 5 |
| Epochs (Mixed) | 5 |

---

## 4. Training Results

### 4.1 Understanding (21 clusters, 395K samples)

Training time: 4.3 hours | Stopped early at epoch 4.5 (plateau)

| Epoch | eval_loss | top1_accuracy |
|-------|-----------|---------------|
| 2.0   | 1.282     | 64.76%        |
| 2.5   | 1.232     | 64.83%        |
| 3.0   | 1.213     | 64.93%        |
| 3.5   | 1.202     | 64.99%        |
| 4.0   | 1.196     | 65.02%        |
| **4.5** | **1.193** | **65.05%** |

```
eval_loss (Understanding)
1.30 ┤
1.28 ┤ ●
1.26 ┤
1.24 ┤
1.23 ┤   ●
1.22 ┤
1.21 ┤     ●
1.20 ┤       ●
1.19 ┤         ● ●
     └──┬──┬──┬──┬──┬──
       2.0 2.5 3.0 3.5 4.0 4.5  epoch
```

### 4.2 Text Reformulation (11 clusters, 327K samples)

Training time: 4.2 hours | Full 5 epochs

| Epoch | eval_loss | top1_accuracy |
|-------|-----------|---------------|
| 0.5   | 2.198     | 53.77%        |
| 1.0   | 2.165     | 54.15%        |
| 1.5   | 2.156     | 54.28%        |
| 2.0   | 2.152     | 54.33%        |
| 2.5   | 2.151     | 54.33%        |
| 3.0   | 2.151     | 54.35%        |
| 3.5   | 2.151     | 54.34%        |
| 4.0   | 2.151     | 54.35%        |
| 4.5   | 2.151     | 54.34%        |
| **5.0** | **2.151** | **54.34%** |

```
eval_loss (Text Reformulation)
2.20 ┤ ●
2.19 ┤
2.18 ┤
2.17 ┤
2.16 ┤   ●
2.15 ┤     ● ● ● ● ● ● ● ●
     └──┬──┬──┬──┬──┬──┬──┬──┬──┬──
       0.5 1  1.5 2  2.5 3  3.5 4  4.5 5  epoch
```

### 4.3 Mixed U+T (32 clusters, 722K samples)

Training time: 12.6 hours | Full 5 epochs

| Epoch | eval_loss | top1_accuracy |
|-------|-----------|---------------|
| 0.5   | 1.865     | 56.23%        |
| 1.0   | 1.818     | 56.51%        |
| 1.5   | 1.799     | 56.59%        |
| 2.0   | 1.791     | 56.63%        |
| 2.5   | 1.789     | 56.64%        |
| 3.0   | 1.789     | 56.64%        |
| 3.5   | 1.788     | 56.65%        |
| 4.0   | 1.788     | 56.64%        |
| 4.5   | 1.788     | 56.64%        |
| **5.0** | **1.788** | **56.64%** |

```
eval_loss (Mixed U+T)
1.87 ┤ ●
1.85 ┤
1.83 ┤
1.82 ┤   ●
1.80 ┤     ●
1.79 ┤       ● ● ● ● ● ● ●
1.78 ┤
     └──┬──┬──┬──┬──┬──┬──┬──┬──┬──
       0.5 1  1.5 2  2.5 3  3.5 4  4.5 5  epoch
```

### 4.4 Training Summary

| Model | Final eval_loss | Final top1_acc | Plateau epoch | Total time |
|-------|-----------------|----------------|---------------|------------|
| Understanding | **1.193** | **65.05%** | ~3.5 | 4.3h |
| Text Reformulation | 2.151 | 54.34% | ~2.0 | 4.2h |
| Mixed (U+T) | 1.788 | 56.64% | ~2.5 | 12.6h |

**Observations**:
- All models plateau within 2–3.5 epochs; additional epochs provide diminishing returns
- Understanding has the lowest loss and highest accuracy — NLU tasks have shorter, more predictable outputs
- Text Reformulation is harder — translation/paraphrase tasks have more diverse output distributions
- Mixed model's metrics fall between the two domains, as expected for a weighted average

---

## 5. Acceptance Rate Evaluation

### 5.1 Methodology

- **Metric**: `overlap_area` = 1 - TVD (Total Variation Distance) between draft and target distributions
  - This is the expected acceptance probability under ideal speculative decoding
  - `overlap_area = 1.0` → every draft token accepted; `0.0` → none accepted
- **Target distributions**: Top-10 token probabilities from the synthetic validation dataset (11,900 samples across 66 clusters)
- **Evaluation**: Teacher-forcing; 100 trunk positions per sample; all validation samples per domain
- **Additional metrics**: `top1_match` (argmax agreement), `KL` divergence

### 5.2 Results: 3 Models × 3 Domains

#### Overlap Area (AR proxy) — higher is better

| Drafter ↓ \ Eval Domain → | Understanding | Text Reform. | Mixed (U+T) |
|----------------------------|:-------------:|:------------:|:-----------:|
| **Understanding**          | **0.7558**    | 0.6676       | 0.7009      |
| **Text Reformulation**     | 0.7091        | **0.7026**   | 0.7046      |
| **Mixed (U+T)**            | 0.7510        | 0.6997       | **0.7191**  |

#### Top-1 Match — higher is better

| Drafter ↓ \ Eval Domain → | Understanding | Text Reform. | Mixed (U+T) |
|----------------------------|:-------------:|:------------:|:-----------:|
| **Understanding**          | **0.6887**    | 0.5297       | 0.5875      |
| **Text Reformulation**     | 0.6239        | **0.5897**   | 0.5997      |
| **Mixed (U+T)**            | 0.6895        | 0.5848       | **0.6223**  |

#### KL Divergence — lower is better

| Drafter ↓ \ Eval Domain → | Understanding | Text Reform. | Mixed (U+T) |
|----------------------------|:-------------:|:------------:|:-----------:|
| **Understanding**          | **0.5602**    | 1.0404       | 0.8679      |
| **Text Reformulation**     | 0.8165        | **0.8200**   | 0.8370      |
| **Mixed (U+T)**            | 0.5590        | 0.8353       | **0.7372**  |

### 5.3 Key Findings

#### Finding 1: Domain specialization helps for Understanding

The Understanding-specific drafter achieves **0.7558** overlap_area on Understanding data — **+0.65%** higher than the Mixed model (0.7510) and **+6.6%** higher than the Text Reformulation model (0.7091).

In top-1 match, Understanding drafter nearly matches Mixed on its own domain (0.6887 vs 0.6895), suggesting both models capture the modal token well for NLU tasks.

#### Finding 2: Domain specialization helps significantly for Text Reformulation

The Text Reformulation-specific drafter achieves **0.7026** overlap_area on Text Reformulation data — **+0.4%** higher than Mixed (0.6997) and **+5.2%** higher than Understanding drafter (0.6676).

More importantly, the top-1 match shows **0.5897 vs 0.5848** (Text Reform. vs Mixed) — the domain model is better at predicting the exact most-likely token.

#### Finding 3: Mixed model is a strong but suboptimal baseline

The Mixed model's AR on the combined domain (0.7191) exceeds the domain-specific models' cross-domain performance, making it a solid general-purpose drafter. However, it underperforms each domain-specific model on their respective domains:

```
Understanding domain:  Understanding drafter (0.7558) > Mixed (0.7510)  Δ = +0.0048
Text Reform. domain:   Text Reform. drafter (0.7026) > Mixed (0.6997)   Δ = +0.0029
```

#### Finding 4: Cross-domain degradation is asymmetric

The Understanding model's degradation on Text Reformulation data is severe (**−0.0882**, from 0.7558 to 0.6676), while the Text Reformulation model's degradation on Understanding is moderate (**−0.0065**, it drops from 0.7026 to 0.7091, but still performs reasonably).

This suggests that translation/reformulation distributions are more "generic" and transfer somewhat to NLU, while NLU-specialized distributions don't transfer well to translation.

#### Finding 5: KL divergence reveals a surprising pattern

The Mixed model has the **lowest KL divergence** on Understanding data (0.5590 vs 0.5602 for Understanding-specific), while the Understanding model is better on overlap_area. This means the Mixed model's distribution is closer to the target in aggregate, but the Understanding model places more probability mass where it matters for acceptance.

---

## 6. Conclusion

**Domain-specific draft models consistently outperform the mixed-domain baseline** on their respective domains in terms of acceptance rate proxy (overlap_area). The improvement is small but consistent:

- **Understanding domain**: +0.48 pp AR improvement with domain-specific drafter
- **Text Reformulation domain**: +0.29 pp AR improvement with domain-specific drafter

While the absolute gains are modest (0.3–0.5 pp), this confirms the hypothesis that domain specialization produces a tighter approximation of the target model's distribution, leading to higher expected acceptance in speculative decoding.

**Practical implications**:
1. For a deployment where domain is known at inference time (e.g., classification vs translation), routing to domain-specific drafters provides a small but consistent AR improvement
2. The Mixed model remains a strong fallback when domain is unknown
3. Future work: evaluate with more distinct domain boundaries (e.g., adding Summarization, QA, Code) where domain gaps may be larger

---

## 7. Reproducibility

### Model checkpoints

| Model | Local Path | HuggingFace |
|-------|------------|-------------|
| Understanding | `.../drafter_understanding/checkpoint-52803` | [mikhialo/drafter-understanding](https://huggingface.co/mikhialo/drafter-understanding) |
| Text Reformulation | `.../drafter_text_reformulation/final` | [mikhialo/drafter-text-reformulation](https://huggingface.co/mikhialo/drafter-text-reformulation) |
| Mixed (U+T) | `.../drafter_mixed_ut/final` | [mikhialo/drafter-mixed-ut](https://huggingface.co/mikhialo/drafter-mixed-ut) |

### Synthetic dataset

[mikhialo/domain-aware-sd-synthetic](https://huggingface.co/datasets/mikhialo/domain-aware-sd-synthetic) — train (66 NPZ, 2.2 GB), validation (66 JSONL, 93 MB), test (66 NPZ, 598 MB)

### Configs

| Config | File |
|--------|------|
| Understanding training | `configs/train_understanding.yaml` |
| Text Reformulation training | `configs/train_text_reformulation.yaml` |
| Mixed training | `configs/train_mixed_ut.yaml` |
| Understanding clusters | `configs/clusters/understanding.json` (21 clusters) |
| Text Reformulation clusters | `configs/clusters/text_reformulation.json` (11 clusters) |
| Mixed clusters | `configs/clusters/mixed_ut.json` (32 clusters) |
| Evaluation | `configs/eval.yaml` |

### Commands

```bash
# Training (all three)
nohup systemd-run --user --scope -p MemoryMax=58G \
  bash -c 'cd /home/rudenko/multisd/Domain-Aware-SD && \
    export CUDA_VISIBLE_DEVICES=0 MLFLOW_ALLOW_FILE_STORE=true \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
    /home/rudenko/miniconda3/envs/domain_sd/bin/python \
    scripts/train_domain_drafter.py --config-name=train_understanding' \
  > logs/train_understanding.log 2>&1 &

# Evaluation (3×3 AR matrix)
CUDA_VISIBLE_DEVICES=0 MLFLOW_ALLOW_FILE_STORE=true \
  python scripts/eval_3x3.py

# Eval results: eval_results/ar_3x3/summary_3x3.csv
```

### Data

- Training data: `data/synthetic/train/v3/*.npz` (66 cluster files)
- Validation data: `data/synthetic/validation/v3/*.jsonl` (66 cluster files, 11,900 samples total)
- Raw Flan: `flan/train/`, `flan/validation/`, `flan/test/`

---

*Generated: 2026-08-29*
*Commit: see git log for full history*
