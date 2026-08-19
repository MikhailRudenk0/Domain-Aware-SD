#!/usr/bin/env bash
# Regenerate all synthetic splits with the RoPE-fixed target model (HF backend).
#
# Order: validation -> test -> train. After the validation split is generated
# it is sanity-checked (mean top-1 prob and mean length); if it looks like the
# old garbage, the script aborts before burning GPU-days on test/train.
#
# Restartable by design:
#   * finished clusters are skipped (output file already exists),
#   * cluster files are written atomically (tmp + rename), so a killed run
#     never leaves a truncated file behind,
#   * each shard retries up to 5 times with a 60 s pause.
#
# Usage:
#   nohup bash scripts/regen_synthetic_all.sh > logs/regen_all.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate domain_sd
export MLFLOW_ALLOW_FILE_STORE=true
mkdir -p logs

VERSION=v3
SHARD_TIMEOUT=172800   # 48 h hang protection per shard attempt

run_split () {
  local split=$1 max=$2 fmt=$3 outdir=$4
  echo "=== SPLIT $split -> $outdir/$VERSION (max=$max fmt=$fmt) $(date) ==="
  for gpu in 0 1; do
    (
      for attempt in 1 2 3 4 5; do
        echo "=== split=$split shard=$gpu attempt=$attempt $(date) ==="
        timeout "$SHARD_TIMEOUT" env CUDA_VISIBLE_DEVICES=$gpu \
          python src/generate_synthetic_data.py \
            data.split="$split" \
            data.max_samples_per_cluster="$max" \
            data.num_shards=2 data.shard_id=$gpu \
            output.format="$fmt" output.dir="$outdir" output.version="$VERSION" \
            s3.upload_after_generation=false \
          && { echo "=== shard $gpu OK $(date) ==="; break; }
        echo "shard $gpu failed (attempt $attempt), retrying in 60 s"
        sleep 60
      done
    ) >> "logs/regen_${split}_shard${gpu}.log" 2>&1 &
  done
  wait
  python src/repro/validate_synthetic.py --split "$outdir/$VERSION" --sample-text 2 \
    > "logs/validate_${split}_${VERSION}.log" 2>&1
  echo "--- validation summary ($split):"
  tail -3 "logs/validate_${split}_${VERSION}.log"
}

check_quality () {
  # abort if the split looks like broken-RoPE garbage
  local logfile=$1
  python - "$logfile" <<'EOF'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"OVERALL: len=([\d.]+) ascii=([\d.]+) rep4=([\d.]+) top1=([\d.]+)", text)
if not m:
    sys.exit("no OVERALL line in " + sys.argv[1])
length, ascii_, rep4, top1 = map(float, m.groups())
ok = top1 > 0.3 and length < 250
print(f"quality gate: len={length} top1={top1} -> {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
EOF
}

run_split validation 200 jsonl data/synthetic/validation
check_quality "logs/validate_validation_${VERSION}.log" || {
  echo "QUALITY GATE FAILED on validation split — aborting before test/train."
  exit 1
}

run_split test null npz data/synthetic/test
run_split train 30000 npz data/synthetic/train

echo "ALL SPLITS DONE $(date)"
