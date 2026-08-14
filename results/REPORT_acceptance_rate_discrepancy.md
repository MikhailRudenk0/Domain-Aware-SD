# Отчёт: расхождение Acceptance Rate (baseline) при перегенерации датасета

## Контекст

На старых графиках (`ar_per_cluster_first100_tokens.jpg`) baseline модель
(`Lite-Mistral-150M-v2-Instruct`, нетренированная) показывала:
- **Baseline mean = 0.480, std = 0.159** (overlap_area, first 100 tokens)
- **SFT mean = 0.713, std = 0.146**

После перегенерации синтетического датасета baseline выдаёт **другие цифры**.

---

## Как работает eval pipeline

### Метрика

```
overlap_area = Σ min(draft_prob, target_prob)
```

Это 1 - TVD (Total Variation Distance), вычисляется на **top-K поддержке target'а**
(обычно K=10). Вероятности draft'а на этих K токенах ренормализуются, чтобы суммировались в 1.

### Два режима target_source

1. **`target_source: model`** — target-модель загружается и прогоняется через teacher-forcing
   на каждом батче. Top-K берётся из полного softmax target'а.

2. **`target_source: dataset`** — top-K target'а берётся из полей `top10_ids`/`top10_probs`
   синтетического датасета (записаны при генерации).

### Входные данные

- При `target_source: model` + данные в формате **flan** (`inputs`/`targets`):
  trunk = токенизированный `targets` (детерминированно).
  Результат **не зависит** от синтетических данных.

- При `target_source: dataset` + данные в формате **synthetic** (`trunk`/`top10_ids`/`top10_probs`):
  trunk = случайно сгенерированная последовательность (sampling из target).
  Результат **полностью зависит** от конкретного датасета.

---

## Анализ причин расхождения

### Гипотеза 1: Старый график строился с `target_source: dataset` (НАИБОЛЕЕ ВЕРОЯТНАЯ)

**Почему это объясняет расхождение:**

Если старый eval использовал `target_source: dataset`:
- Target top-K бралось из записанных `top10_ids`/`top10_probs` в synthetic v1
- Trunk — из поля `trunk` (случайно сгенерированный при sampling)
- При **перегенерации** датасета изменились и trunk, и top10 → **другие** входы для eval
- Baseline модель видит другие контексты → другие вероятности → другой overlap_area

Если же eval использовал `target_source: model`:
- Target top-K вычисляется заново через forward pass модели
- Trunk из flan validation (детерминированный tokenized `targets`)
- Результат **не должен** зависеть от перегенерации датасета

**Текущий конфиг** (`configs/eval.yaml`):
```yaml
target_source: model
datasets:
  - data/validation/v1    # НЕ СУЩЕСТВУЕТ ни локально, ни на S3!
```

**Проблема:** `data/validation/v1` — несуществующий путь. На S3 есть только:
- `data/synthetic/v1/` — синтетические данные (JSONL с trunk + top10)
- `data/synthetic/train/v2/` — NPZ тренировочные данные
- `data/synthetic/test/v1/` — NPZ тестовые данные

Вероятно, для старого эксперимента `data/validation/v1` был симлинком или копией
`data/synthetic/v1/`, а eval работал с `target_source: dataset`.

**Проверить**: если поставить `target_source: dataset` и `datasets: data/synthetic/v1`,
то eval будет использовать top-K из датасета. Новый датасет (v2) даст другие top-K →
другой overlap_area.

### Гипотеза 2: Разный drafter

`Lite-Mistral-150M-v2-Instruct` — конкретная модель от OuteAI (150M параметров,
12 слоёв, hidden_size=768). Она загружена и на S3, и локально. Vocab size = 32064.

Если в прошлый раз использовался **другой** drafter (например, `tiny-mixtral` с vocab_size=32000),
результаты будут другими. Но в конфиге `eval.yaml` чётко указан `Lite-Mistral-150M-v2-Instruct`.

**Проверить**: какая модель реально использовалась. Если eval JSON-ы не сохранились — невозможно
установить точно.

### Гипотеза 3: Ренормализация top-K → разный K

В `_build_inputs_single` (evaluator.py:62-81):
```python
draft_at_target = di.prob_at_target_topk  # draft probs at target's top-K ids
s = float(draft_at_target.sum())
draft_aligned = draft_at_target / s       # РЕНОРМАЛИЗАЦИЯ!
```

Overlap_area вычисляется **на ренормализованных** распределениях в пределах K.
Если target имеет sharp distribution (одна вероятность ~0.9), то K=10 покрывает
почти весь probability mass, и ренормализация минимальна. Но если target распределён
равномерно, то top-10 может покрывать только 30-40% mass, и ренормализация
сильно искажает.

При разном датасете — разное распределение сложности позиций → разный эффект ренормализации.

### Гипотеза 4: `skip_top10_above_prob` фильтрация

При генерации синтетических данных позиции с `top1_prob > skip_top10_above_prob`
пропускаются (пустой top10). В eval эти позиции помечаются как `is_skipped` и
**исключаются** из подсчёта метрик.

В `generation.yaml`:
```yaml
skip_top10_above_prob: 1.0
```

Значение 1.0 означает "никогда не пропускать" — все позиции имеют top10. Но если при
старой генерации значение было другим (например, 0.95), то «лёгкие» позиции (где
target очень уверен) пропускались → baseline на оставшихся «сложных» позициях мог
показывать **другой** средний overlap_area.

### Гипотеза 5: round(..., 3) квантизация вероятностей

В `SpecDecDataset.__getitem__` (dataset.py:178):
```python
pos_probs = [round(float(p), 3) for p in pos_probs][:n]
```

Вероятности округляются до 3 знаков. При `target_source: dataset` это может
вносить ≈0.0005 ошибку на каждую позицию. На 100 позициях × 10 top-K entries
это незначительно (~0.005 суммарно), но в сочетании с ренормализацией эффект
может усиливаться.

---

## Рекомендации

### 1. Определить, какой `target_source` использовался на старом графике

Это **главный** вопрос. Два способа проверить:
- Найти старые eval JSON-ы (если сохранились где-то)
- Запустить eval с обоими вариантами на GPU:
  - `target_source: dataset` + `data/synthetic/v1` → сравнить с графиком
  - `target_source: model` + `flan/validation/` → сравнить с графиком

### 2. Проверить на GPU с target_source: model + flan validation

```bash
python src/eval/main.py \
  target_source=model \
  'datasets=[flan/validation]' \
  draft_models='[Lite-Mistral-150M-v2-Instruct]' \
  device=cuda
```

Этот результат **не зависит** от синтетических данных. Если он совпадает с графиком —
значит старый eval тоже использовал `target_source: model`, и проблема в другом.

### 3. Проверить на GPU с target_source: dataset + old synthetic v1

```bash
# Скачать data/synthetic/v1/ с S3
python src/download_from_s3.py --data --version v1

python src/eval/main.py \
  target_source=dataset \
  'datasets=[data/synthetic/v1]' \
  draft_models='[Lite-Mistral-150M-v2-Instruct]' \
  device=cuda
```

Если результат совпадает с графиком — значит старый eval использовал `target_source: dataset`.
Тогда расхождение объясняется тем, что **новый датасет имеет другие trunk и top-K**.

### 4. Если проблема в target_source: dataset

Это **не баг** — это **ожидаемое поведение**. Разные сгенерированные trunk дают разные
overlap_area. Для сравнимых результатов нужно либо:
- Использовать `target_source: model` + фиксированные flan validation данные
- Использовать **один и тот же** синтетический датасет для всех сравнений

---

## Ограничения текущего анализа

- **Нет GPU** на этом сервере → не могу запустить eval и проверить гипотезы экспериментально
- **Нет старых eval JSON-ов** → не могу точно установить, какие параметры использовались
- **Конфиг мог быть изменён** между экспериментами (overrides через CLI Hydra не сохранены)

---

## Вывод

Наиболее вероятная причина расхождения: **старый eval использовал `target_source: dataset`
с синтетическими данными v1, а после перегенерации trunk и top-K изменились, что привело
к другим значениям overlap_area для baseline**.

Для baseline модели (нетренированной) overlap_area сильно зависит от:
1. Конкретных trunk tokens (контекст, на котором draft делает prediction)
2. Конкретного target top-K (с чем сравнивается draft)

Оба этих фактора меняются при перегенерации синтетических данных.
