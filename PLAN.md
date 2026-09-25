# План: обучение 5 per-cluster драфтеров (WMT16 translation)

## Цель
Обучить 5 отдельных драфтеров — по одному на каждый из 5 кластеров WMT16 translation, которые показали наименьший AR на графике:

1. `wmt16_translate_ruen_10templates` (30k samples, 71MB)
2. `wmt16_translate_csen_10templates` (30k samples, 47MB)
3. `wmt16_translate_fien_10templates` (30k samples, 62MB)
4. `wmt16_translate_tren_10templates` (30k samples, 68MB)
5. `wmt16_translate_deen_10templates` (30k samples, 55MB)

## Инфраструктура
- 2x NVIDIA RTX 3090 (24 GB каждая), обе свободны
- Обучаем по 2 модели параллельно (GPU 0 + GPU 1)
- Волна 1: ruen (GPU 0) + csen (GPU 1)
- Волна 2: fien (GPU 0) + tren (GPU 1)
- Волна 3: deen (GPU 0)

## Гиперпараметры
Те же, что в предыдущих тренировках, с адаптацией для маленького датасета:

| Параметр | Значение | Комментарий |
|----------|----------|-------------|
| Базовая модель | Lite-Mistral-150M-v2-Instruct | 156M params |
| dtype | bfloat16 | |
| Batch size | 32 | |
| LR | 5e-5 | |
| LR schedule | cosine + 3% warmup | |
| Loss | mixed (0.5 CE + 0.5 KD, T=1.0) | |
| Эпохи | 10 (макс) | С ранней остановкой |
| Val fraction | 0.05 | ~1500 samples |
| Save/eval | 2x per epoch | ~469 steps (30000*0.95/32/2) |
| save_total_limit | 4 | |
| max_length | 512 | |
| max_gen_length | 256 | |

## Ожидаемые метрики
- Каждый кластер: ~30k samples → ~28,500 train, ~1,500 val
- Steps per epoch: ~891 (28500 / 32)
- Save/eval every: ~445 steps
- Скорость: ~3.4 it/s → ~4.4 мин/эпоха → ~44 мин на 10 эпох
- Ожидаемое плато: эпоха 2-4 (по аналогии с Text Reformulation)

## Признаки переобучения (критерии остановки)
- eval_loss растёт 2 чекпоинта подряд (1 эпоху)
- top1_accuracy на валидации падает
- train_loss продолжает падать при растущем eval_loss
- Если плато > 3 эпох — остановить

## Что нужно создать
1. 5 конфигов: `configs/train_wmt16_translate_{lang}.yaml`
2. 5 JSON-файлов кластеров: `configs/clusters/wmt16_translate_{lang}.json`
3. Shell-скрипт для запуска волнами: `scripts/run_wmt16_training.sh`

## Мониторинг
- Первые 5 минут: проверить что обучение стартовало, loss падает
- Каждые 5 минут первые 30 мин: проверять eval_loss и top1_accuracy
- Затем каждый час: проверять кривые обучения
- При обнаружении переобучения: немедленно остановить процесс

## Чекпоинты и результаты
- Output: `/media/public/rudenko/projects/Domain-Aware-SD/outputs/drafter_wmt16_translate_{lang}/`
- Логи: `logs/train_wmt16_translate_{lang}.log`
- MLflow: `./mlruns`, experiment `domain_drafters`
- По окончании: сохранить лучший чекпоинт как `final/`
- Составить отчёт с таблицами eval_loss / top1_accuracy по эпохам
