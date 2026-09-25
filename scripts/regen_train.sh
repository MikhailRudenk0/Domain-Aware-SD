#!/usr/bin/env bash
# Generate train/v3 split (test and validation already done)
set -u
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate domain_sd
export MLFLOW_ALLOW_FILE_STORE=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

VERSION=v3

echo "=== SPLIT train -> data/synthetic/train/$VERSION $(date) ==="
rm -f logs/FAILED_train_*
for gpu in 0 1; do
  (
    ok=0
    for attempt in 1 2 3 4 5; do
      echo "=== split=train shard=$gpu attempt=$attempt $(date) ==="
      timeout 172800 env CUDA_VISIBLE_DEVICES=$gpu \
        python src/generate_synthetic_data.py \
          data.split=train \
          data.max_samples_per_cluster=30000 \
          data.num_shards=2 data.shard_id=$gpu \
          output.format=npz output.dir=data/synthetic/train output.version=$VERSION \
          s3.upload_after_generation=false \
        && { echo "=== shard $gpu OK $(date) ==="; ok=1; break; }
      echo "shard $gpu failed (attempt $attempt), retrying in 60 s"
      sleep 60
    done
    [ "$ok" = 1 ] || touch "logs/FAILED_train_${gpu}"
  ) >> "logs/regen_train_shard${gpu}.log" 2>&1 &
done
wait

if ls logs/FAILED_train_* >/dev/null 2>&1; then
  echo "SHARD FAILURE in train split"
  exit 1
fi

python src/repro/validate_synthetic.py --split data/synthetic/train/$VERSION --sample-text 2 \
  > "logs/validate_train_${VERSION}.log" 2>&1
echo "--- validation summary (train):"
tail -5 "logs/validate_train_${VERSION}.log"
echo "=== TRAIN DONE $(date) ==="
