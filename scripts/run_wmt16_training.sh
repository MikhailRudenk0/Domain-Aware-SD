#!/usr/bin/env bash
# ============================================================================
# Per-cluster drafter training: 5 WMT16 translation clusters
#
# Trains drafters in 3 WAVES on 2 GPUs:
#   Wave 1: ruen (GPU 0) + csen (GPU 1)
#   Wave 2: fien (GPU 0) + tren (GPU 1)
#   Wave 3: deen (GPU 0)
#
# Each run:
#   - 10 epochs max, mixed loss (0.5 CE + 0.5 KD)
#   - Checkpoints saved twice per epoch (auto-calculated)
#   - Auto-restart on failure (up to 5 attempts, resumes from last checkpoint)
#   - Full logs in logs/train_wmt16_translate_<lang>.log
#
# Usage:
#   bash scripts/run_wmt16_training.sh              # all 3 waves sequentially
#   bash scripts/run_wmt16_training.sh wave1         # only wave 1 (ruen + csen)
#   bash scripts/run_wmt16_training.sh wave2         # only wave 2 (fien + tren)
#   bash scripts/run_wmt16_training.sh wave3         # only wave 3 (deen)
#   bash scripts/run_wmt16_training.sh ruen          # single cluster
#
# Prerequisites: GPU host with conda env domain_sd, 2x GPUs
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

# ── Environment ──────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate domain_sd
export MLFLOW_ALLOW_FILE_STORE=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

# ── Train one cluster ────────────────────────────────────────────────────────

train_one_cluster() {
    local cluster="$1"
    local gpu="$2"
    local config_name="train_wmt16_translate_${cluster}"
    local log_file="logs/train_wmt16_translate_${cluster}.log"
    local max_attempts=5

    echo "============================================================" | tee -a "$log_file"
    echo "=== Training: wmt16_translate_${cluster} on GPU:${gpu} ($(date)) ===" | tee -a "$log_file"
    echo "============================================================" | tee -a "$log_file"

    local ok=0
    for attempt in $(seq 1 $max_attempts); do
        echo "" | tee -a "$log_file"
        echo "--- Attempt ${attempt}/${max_attempts} ($(date)) ---" | tee -a "$log_file"

        if CUDA_VISIBLE_DEVICES=${gpu} python scripts/train_domain_drafter.py \
            --config-name="${config_name}" \
            >> "$log_file" 2>&1; then
            echo "=== wmt16_translate_${cluster} COMPLETED SUCCESSFULLY ($(date)) ===" | tee -a "$log_file"
            ok=1
            break
        else
            local exit_code=$?
            echo "[error] Training failed with exit code ${exit_code}" | tee -a "$log_file"

            if grep -q "CUDA out of memory\|OutOfMemoryError" "$log_file" 2>/dev/null; then
                echo "[error] OOM detected on GPU:${gpu}!" | tee -a "$log_file"
            fi

            if [ "$attempt" -lt "$max_attempts" ]; then
                echo "[retry] Waiting 30s before retry (will resume from checkpoint)..." | tee -a "$log_file"
                sleep 30
            fi
        fi
    done

    if [ "$ok" -eq 0 ]; then
        echo "!!! FAILED after ${max_attempts} attempts: wmt16_translate_${cluster} !!!" | tee -a "$log_file"
        touch "logs/FAILED_wmt16_translate_${cluster}"
        return 1
    fi

    return 0
}

# ── Wave functions ───────────────────────────────────────────────────────────

run_wave1() {
    echo ">>> WAVE 1: ruen (GPU 0) + csen (GPU 1) — $(date)"
    train_one_cluster "ruen" 0 &
    local PID0=$!
    train_one_cluster "csen" 1 &
    local PID1=$!

    local FAIL=0
    wait $PID0 || FAIL=1
    wait $PID1 || FAIL=1

    if [ "$FAIL" -ne 0 ]; then
        echo "!!! WAVE 1 had failures — check logs/" >&2
        return 1
    fi
    echo ">>> WAVE 1 COMPLETE — $(date)"
}

run_wave2() {
    echo ">>> WAVE 2: fien (GPU 0) + tren (GPU 1) — $(date)"
    train_one_cluster "fien" 0 &
    local PID0=$!
    train_one_cluster "tren" 1 &
    local PID1=$!

    local FAIL=0
    wait $PID0 || FAIL=1
    wait $PID1 || FAIL=1

    if [ "$FAIL" -ne 0 ]; then
        echo "!!! WAVE 2 had failures — check logs/" >&2
        return 1
    fi
    echo ">>> WAVE 2 COMPLETE — $(date)"
}

run_wave3() {
    echo ">>> WAVE 3: deen (GPU 0) — $(date)"
    train_one_cluster "deen" 0
    echo ">>> WAVE 3 COMPLETE — $(date)"
}

# ── Main ─────────────────────────────────────────────────────────────────────

FILTER="${1:-all}"

echo "======================================"
echo "WMT16 per-cluster drafter training"
echo "Started: $(date)"
echo "Filter: ${FILTER}"
echo "======================================"

FAILED=0

case "$FILTER" in
    all)
        run_wave1 || FAILED=1
        run_wave2 || FAILED=1
        run_wave3 || FAILED=1
        ;;
    wave1) run_wave1 || FAILED=1 ;;
    wave2) run_wave2 || FAILED=1 ;;
    wave3) run_wave3 || FAILED=1 ;;
    ruen)  train_one_cluster "ruen" 0 || FAILED=1 ;;
    csen)  train_one_cluster "csen" 1 || FAILED=1 ;;
    fien)  train_one_cluster "fien" 0 || FAILED=1 ;;
    tren)  train_one_cluster "tren" 1 || FAILED=1 ;;
    deen)  train_one_cluster "deen" 0 || FAILED=1 ;;
    *)
        echo "Unknown filter: ${FILTER}"
        echo "Usage: $0 [all|wave1|wave2|wave3|ruen|csen|fien|tren|deen]"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "All training finished: $(date)"
if [ "$FAILED" -eq 0 ]; then
    echo "STATUS: SUCCESS"
else
    echo "STATUS: SOME RUNS FAILED — check logs/"
fi
echo "======================================"

exit $FAILED
