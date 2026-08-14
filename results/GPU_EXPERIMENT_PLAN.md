# План воспроизведения Acceptance Rate на GPU

## Цель

Воспроизвести результаты со старого графика `ar_per_cluster_first100_tokens.jpg`:
- **Baseline mean = 0.480, std = 0.159** (overlap_area, first 100 tokens)
- **SFT (general) mean = 0.713, std = 0.146**

Конкретно — воспроизвести **baseline** (нетренированная модель), т.к. SFT требует
обучения на тех же данных.

Нас особенно интересует **first 1 token** (position 0), т.к. для него нужно только
распределение после первого токена — и можно надеяться на идентичное воспроизведение
(один forward pass, детерминированный softmax).

---

## Известные факты

1. **target_source: model** — target top-K считается через forward pass TurboSparse
2. **Старая синтетика потеряна** — не можем точно узнать параметры генерации
3. **Pipeline написан с нуля** — код полностью новый
4. **Drafter на старом графике**: `Lite-Mistral-150M-v2-Instruct` (150M, OuteAI)
5. **Target**: `TurboSparse-Mistral-Instruct` (7B sparsified Mixtral, trust_remote_code)
6. **Данные для eval**: вероятно flan validation (66 кластеров × 30-200 сэмплов = 11930 всего)

---

## Сервер: ограничения

- VPS: 2 ГБ RAM, нет GPU
- Target модель 15 ГБ — не помещается даже в RAM
- CPU инференс drafter'а: ~1 сек/сэмпл (feasible для drafter alone, ~3.3 часа)
- Вывод: **нужна машина с GPU (≥24 ГБ VRAM для target + drafter одновременно)**

---

## Что нужно на GPU-машине

### Минимальные требования
- GPU: ≥24 ГБ VRAM (A100 40GB идеально, A10 24GB может хватить с bfloat16)
- RAM: ≥32 ГБ
- Диск: ≥50 ГБ свободного места

### Файлы для загрузки
Всё уже на S3 `s3://domain-aware-sd` (публичное чтение без кредов):
```bash
# Скачать модели
python src/download_from_s3.py --models

# Скачать flan validation (если нет)
# Из Hugging Face или скопировать flan/validation/ с этого сервера
```

### Зависимости
```bash
conda env create -f environment.yml
# или
pip install torch transformers hydra-core omegaconf mlflow python-dotenv boto3 tqdm matplotlib
```

---

## Эксперименты

### Эксперимент 1: Базовый — текущий конфиг

Проверить, что pipeline работает и какие цифры даёт.

```bash
python src/eval/main.py \
  target_source=model \
  'datasets=[flan/validation]' \
  'draft_models=[Lite-Mistral-150M-v2-Instruct]' \
  n_positions=100 \
  batch_size=8 \
  device=cuda \
  dtype=bfloat16
```

Ожидаемый результат: overlap_area per cluster per position → JSON в eval_results/.

### Эксперимент 2: Перебор dtype target'а

Dtype может влиять на softmax → разное top-K target → разный overlap_area.

```bash
# bfloat16 (default)
python src/eval/main.py dtype=bfloat16 device=cuda mlflow.run_name=bf16

# float16
python src/eval/main.py dtype=float16 device=cuda mlflow.run_name=fp16

# float32 (самый точный, но нужно больше VRAM)
python src/eval/main.py dtype=float32 device=cuda mlflow.run_name=fp32
```

**Почему это важно:** TurboSparse — sparsified model с кастомным кодом
(`modeling_bamboo.py`). Разные dtype могут давать заметно разные softmax,
особенно при больших логитах. Если старый eval использовал float32, а новый
bfloat16 — результаты будут отличаться.

### Эксперимент 3: Перебор topk_K

Top-K target'а определяет, на какой поддержке считается overlap_area.

```bash
# K=10 (default)
python src/eval/main.py topk_K=10 mlflow.run_name=topk10

# K=20
python src/eval/main.py topk_K=20 mlflow.run_name=topk20

# K=50
python src/eval/main.py topk_K=50 mlflow.run_name=topk50

# K=100 (ближе к полному словарю)
python src/eval/main.py topk_K=100 mlflow.run_name=topk100
```

**Почему:** overlap_area = Σ min(draft, target) на K-поддержке с ренормализацией.
Больший K → больше хвостовых токенов → baseline overlap может расти
(длинный хвост с маленькими вероятностями, где draft тоже имеет маленькие).

### Эксперимент 4: Другой drafter — tiny-mixtral

```bash
python src/eval/main.py \
  'draft_models=[tiny-mixtral]' \
  device=cuda \
  mlflow.run_name=tiny-mixtral-baseline
```

**Почему:** если на старом графике использовался `tiny-mixtral` (а не
`Lite-Mistral`), то это объяснит расхождение. tiny-mixtral имеет:
- vocab_size=32000 (vs 32768 у Lite-Mistral → 768 невалидных target токенов)
- ~942 МБ (vs 597 МБ)
- Другая архитектура (MoE vs dense)

### Эксперимент 5: Оба drafter'а side-by-side

```bash
python src/eval/main.py \
  'draft_models=[Lite-Mistral-150M-v2-Instruct]' \
  device=cuda \
  mlflow.run_name=lite-mistral-baseline

python src/eval/main.py \
  'draft_models=[tiny-mixtral]' \
  device=cuda \
  mlflow.run_name=tiny-mixtral-baseline
```

Затем визуализировать:
```bash
python src/viz/acceptance_bar.py \
  position='[0,99]' \
  labels='{Lite-Mistral-150M-v2-Instruct: lite-mistral, tiny-mixtral: tiny-mixtral}' \
  title='Acceptance Rate per Cluster (first 100 tokens)'
```

### Эксперимент 6: Только first 1 token (position 0)

Самый быстрый тест — position 0 детерминирован и не зависит от авторегрессии.

```bash
python src/eval/main.py \
  n_positions=1 \
  device=cuda \
  mlflow.run_name=first1token

# Визуализация
python src/viz/acceptance_bar.py \
  position=0 \
  title='Acceptance Rate per Cluster (first 1 token)'
```

Сравнить с `ar_per_cluster_first1_token.jpg`:
- SFT mean=0.809, std=0.151
- Baseline mean=0.373, std=0.181

### Эксперимент 7: Влияние tokenizer'а

В `main.py` (строка 72-74) используется **target tokenizer** для всей токенизации:
```python
tokenizer = AutoTokenizer.from_pretrained(
    str(target_dir), trust_remote_code=cfg.target.trust_remote_code
)
```

TurboSparse tokenizer имеет vocab_size=32064, а Lite-Mistral — 32768.
Draft model обрезает target'овые ID ≥ 32768 (safe clamping), но target'овые
ID в диапазоне 32000-32063 (специальные токены TurboSparse) могут попадать
в top-K target'а и не иметь осмысленного значения в draft'е.

**Проверить:** сколько позиций помечается как `n_special_target_per_position`
и `n_samples_with_oob_in_trunk`. Если много — это может сильно менять
средний overlap_area (эти позиции исключаются из подсчёта).

---

## Независимый скрипт проверки (minimal)

Если есть сомнения в корректности pipeline, вот минимальный скрипт,
который считает overlap_area с нуля, без зависимостей от eval/:

```python
#!/usr/bin/env python3
"""
Минимальный скрипт проверки overlap_area на одном кластере.
Не зависит от src/eval/ — всё inline.

Usage:
    python verify_overlap.py \
        --cluster aeslc_10templates \
        --max-samples 50 \
        --dtype bfloat16
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_overlap_area_position0(
    target_dir: str,
    draft_dir: str,
    flan_file: str,
    max_samples: int,
    dtype_str: str,
    topk_K: int = 10,
):
    """Compute overlap_area at trunk position 0 for each sample."""

    # Resolve dtype
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[dtype_str]

    # Load tokenizer (target's, as in the main pipeline)
    tokenizer = AutoTokenizer.from_pretrained(target_dir, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id or 0

    # Load models
    print(f"Loading target: {target_dir}")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_dir, trust_remote_code=True, dtype=torch_dtype
    ).cuda().eval()

    print(f"Loading draft: {draft_dir}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_dir, dtype=torch_dtype
    ).cuda().eval()
    draft_vocab = draft_model.config.vocab_size

    # Load flan samples
    records = []
    with open(flan_file) as f:
        for line in f:
            records.append(json.loads(line))
            if len(records) >= max_samples:
                break
    print(f"Loaded {len(records)} samples from {flan_file}")

    overlaps = []
    skipped = 0

    for i, rec in enumerate(records):
        prompt_ids = tokenizer.encode(
            rec["inputs"], add_special_tokens=True, truncation=True, max_length=2047
        )
        trunk_ids = tokenizer.encode(rec["targets"], add_special_tokens=False)
        if not trunk_ids:
            skipped += 1
            continue

        # We only need position 0 → trunk_ids[:1]
        full_ids = prompt_ids + trunk_ids[:1]
        input_ids = torch.tensor([full_ids], dtype=torch.long).cuda()
        attn_mask = torch.ones_like(input_ids)

        gen_start = len(prompt_ids)

        with torch.no_grad():
            # Target forward: logits at gen_start-1 predict position gen_start
            t_logits = target_model(input_ids=input_ids, attention_mask=attn_mask).logits
            t_probs = torch.softmax(t_logits[0, gen_start - 1].float(), dim=-1)
            t_topk = torch.topk(t_probs, k=topk_K)
            t_topk_ids = t_topk.indices.cpu().numpy().astype(np.int64)
            t_topk_probs = t_topk.values.cpu().numpy().astype(np.float32)

            # Skip if target argmax is outside draft vocab
            if int(t_topk_ids[0]) >= draft_vocab:
                skipped += 1
                continue

            # Renormalize target top-K
            t_sum = t_topk_probs.sum()
            if t_sum > 0:
                t_topk_probs = t_topk_probs / t_sum

            # Draft forward: need to clamp OOB token IDs
            input_ids_safe = input_ids.clone()
            input_ids_safe[input_ids_safe >= draft_vocab] = 0

            d_logits = draft_model(input_ids=input_ids_safe, attention_mask=attn_mask).logits
            d_probs = torch.softmax(d_logits[0, gen_start - 1].float(), dim=-1)

            # Extract draft probs at target's top-K ids
            safe_ids = np.where(
                (t_topk_ids >= 0) & (t_topk_ids < draft_vocab), t_topk_ids, 0
            )
            d_at_target = d_probs[torch.tensor(safe_ids, device=d_probs.device)]
            d_at_target = d_at_target.cpu().numpy().astype(np.float32)

            # Zero out OOB positions
            oob = (t_topk_ids < 0) | (t_topk_ids >= draft_vocab)
            d_at_target[oob] = 0.0

            # Renormalize draft at target support
            d_sum = d_at_target.sum()
            if d_sum > 0:
                d_aligned = d_at_target / d_sum
            else:
                d_aligned = np.full(topk_K, 1.0 / topk_K, dtype=np.float32)

            # overlap_area = sum min(draft, target)
            oa = float(np.minimum(d_aligned, t_topk_probs).sum())
            overlaps.append(oa)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(records)}: running mean={np.mean(overlaps):.4f}")

    mean_oa = np.mean(overlaps) if overlaps else 0.0
    std_oa = np.std(overlaps) if overlaps else 0.0
    print(f"\nResult: overlap_area at position 0")
    print(f"  mean={mean_oa:.4f}, std={std_oa:.4f}")
    print(f"  samples={len(overlaps)}, skipped={skipped}")
    return mean_oa, std_oa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="TurboSparse-Mistral-Instruct")
    parser.add_argument("--draft", default="Lite-Mistral-150M-v2-Instruct")
    parser.add_argument("--cluster", default="aeslc_10templates")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--topk-K", type=int, default=10)
    args = parser.parse_args()

    flan_file = f"flan/validation/{args.cluster}_validation.jsonl"
    compute_overlap_area_position0(
        target_dir=args.target,
        draft_dir=args.draft,
        flan_file=flan_file,
        max_samples=args.max_samples,
        dtype_str=args.dtype,
        topk_K=args.topk_K,
    )
```

---

## Чек-лист для GPU

### Подготовка
- [ ] Клонировать репозиторий: `git clone git@github.com:MikhailRudenk0/Domain-Aware-SD.git`
- [ ] Создать `.env` из `.env.example`, заполнить `S3_SECRET_KEY`
- [ ] Скачать модели: `python src/download_from_s3.py --models`
- [ ] Скачать flan validation (скопировать `flan/validation/` или скачать отдельно)
- [ ] Установить зависимости: `pip install -r requirements.txt` или `conda env create`

### Основные эксперименты
- [ ] Эксперимент 1: базовый eval (bfloat16, topk_K=10, Lite-Mistral)
- [ ] Эксперимент 2: dtype sweep (bfloat16 / float16 / float32)
- [ ] Эксперимент 3: topk_K sweep (10 / 20 / 50 / 100)
- [ ] Эксперимент 4: tiny-mixtral как drafter
- [ ] Эксперимент 6: only position 0 (first 1 token) — быстрый тест

### Анализ
- [ ] Сравнить baseline mean/std с reference: 0.480±0.159 (100 tokens), 0.373±0.181 (1 token)
- [ ] Проверить n_special_target_per_position и n_samples_with_oob_in_trunk
- [ ] Построить графики: `python src/viz/acceptance_bar.py`
- [ ] Если ни одна комбинация не даёт 0.480 — проблема в pipeline или в метрике

---

## Возможные причины расхождения (приоритет)

### 1. dtype target'а (ВЫСОКИЙ)
BFloat16 vs float32 softmax даёт разные top-K и разные вероятности.
Для sparsified model разница может быть существенной.

### 2. topk_K (СРЕДНИЙ)
K=10 vs K=20 vs K=50 → разный overlap_area из-за ренормализации.
Чем больше K, тем больше хвост → baseline может расти.

### 3. Vocab mismatch / OOB handling (СРЕДНИЙ)
TurboSparse vocab=32064, Lite-Mistral vocab=32768, tiny-mixtral vocab=32000.
Токены 32000-32063 (специальные TurboSparse) попадают в trunk и в target top-K.
Для tiny-mixtral они все OOB → позиции пропускаются → другой средний overlap.
Для Lite-Mistral они внутри vocab (< 32768) → не пропускаются, но у draft'а
нет осмысленной вероятности для этих токенов.

### 4. Другой drafter (СРЕДНИЙ)
Если на старом графике использовался `tiny-mixtral`, а не `Lite-Mistral`.

### 5. Ренормализация draft на target support (НИЗКИЙ)
Текущий pipeline ренормализует draft probs на top-K target'а. Если в прошлом
eval использовал другую нормализацию (или полный vocab overlap) — результаты
будут отличаться.

### 6. Tokenizer mismatch (НИЗКИЙ)
Prompt encoding через target tokenizer (ChatML template) vs draft tokenizer
(Mistral chat template). Если chat templates разные, то prompt_ids разные →
разный контекст → разные вероятности.

---

## Reference значения для сравнения

### First 1 token (position 0)
- SFT mean=0.809, std=0.151
- Baseline mean=0.373, std=0.181

### First 100 tokens (positions 0-99)
- SFT mean=0.713, std=0.146
- Baseline mean=0.480, std=0.159

### Распределение по task types
- Text Reformulation (translation): самые низкие значения (~0.30-0.45)
- Content Generation (summarization): средние (~0.50-0.70)
- Reasoning (QA, NLI): средне-высокие (~0.60-0.80)
- Understanding (classification): самые высокие (~0.75-0.90)

---

## Файлы проекта

| Файл | Назначение |
|------|------------|
| `src/eval/main.py` | Entry point eval pipeline (Hydra) |
| `src/eval/evaluator.py` | Core eval loop |
| `src/eval/metrics.py` | overlap_area, top1_match, topk_overlap, kl |
| `src/eval/draft_runner.py` | Draft model forward pass |
| `src/eval/target_provider.py` | Model/Dataset target top-K |
| `src/viz/acceptance_bar.py` | Bar chart visualization |
| `configs/eval.yaml` | Eval config |
| `configs/viz_acceptance_bar.yaml` | Viz config |
| `results/reference_ar_first100_tokens.txt` | Распознанные значения со старого графика |
| `results/ar_per_cluster_first100_tokens.jpg` | Старый график (reference) |
| `results/ar_per_cluster_first1_token.jpg` | Старый график first 1 token |
| `results/verify_overlap.py` | Минимальный независимый скрипт проверки |
