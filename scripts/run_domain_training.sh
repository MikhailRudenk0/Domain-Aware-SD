#!/usr/bin/env bash
# ============================================================================
# Domain-specific drafter training: Understanding + Text Reformulation
#
# Trains two drafters IN PARALLEL on separate GPUs. Each run:
#   - 25 epochs, mixed loss (0.5 CE + 0.5 KD) on top-10 target distribution
#   - Checkpoints saved twice per epoch (auto-calculated)
#   - Extended logging (every 10 steps) + eval every half-epoch
#   - Auto-restart on failure (up to 5 attempts, resumes from last checkpoint)
#   - Full logs in logs/train_<domain>.log
#
# Usage:
#   bash scripts/run_domain_training.sh                    # both in parallel
#   bash scripts/run_domain_training.sh understanding      # single domain
#   bash scripts/run_domain_training.sh text_reformulation
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

# ── GPU assignment ───────────────────────────────────────────────────────────
# When running both domains in parallel:
#   Understanding      → GPU 0
#   Text Reformulation → GPU 1
declare -A DOMAIN_GPU
DOMAIN_GPU[understanding]=0
DOMAIN_GPU[text_reformulation]=1

# ── Train one domain ─────────────────────────────────────────────────────────

train_one_domain() {
    local domain="$1"
    local gpu="${DOMAIN_GPU[$domain]:-0}"
    local config_name="train_${domain}"
    local log_file="logs/train_${domain}.log"
    local max_attempts=5

    echo "============================================================" | tee -a "$log_file"
    echo "=== Training: ${domain} on GPU:${gpu} ($(date)) ===" | tee -a "$log_file"
    echo "============================================================" | tee -a "$log_file"

    local ok=0
    for attempt in $(seq 1 $max_attempts); do
        echo "" | tee -a "$log_file"
        echo "--- Attempt ${attempt}/${max_attempts} ($(date)) ---" | tee -a "$log_file"

        if CUDA_VISIBLE_DEVICES=${gpu} python scripts/train_domain_drafter.py \
            --config-name="${config_name}" \
            >> "$log_file" 2>&1; then
            echo "=== ${domain} COMPLETED SUCCESSFULLY ($(date)) ===" | tee -a "$log_file"
            ok=1
            break
        else
            local exit_code=$?
            echo "[error] Training failed with exit code ${exit_code}" | tee -a "$log_file"

            # Check for OOM — log it
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
        echo "!!! FAILED after ${max_attempts} attempts: ${domain} !!!" | tee -a "$log_file"
        touch "logs/FAILED_${domain}"
        return 1
    fi

    return 0
}

# ── Main ─────────────────────────────────────────────────────────────────────

DOMAIN_FILTER="${1:-all}"

echo "======================================"
echo "Domain drafter training"
echo "Started: $(date)"
echo "Filter: ${DOMAIN_FILTER}"
echo "======================================"

FAILED=0

if [ "$DOMAIN_FILTER" = "all" ]; then
    # Run BOTH domains in parallel on separate GPUs
    echo "Launching Understanding on GPU:0 and Text Reformulation on GPU:1 in parallel..."

    train_one_domain "understanding" &
    PID_UNDERSTANDING=$!

    train_one_domain "text_reformulation" &
    PID_REFORMULATION=$!

    # Wait for both
    FAIL_U=0
    FAIL_R=0
    wait $PID_UNDERSTANDING || FAIL_U=1
    wait $PID_REFORMULATION || FAIL_R=1

    if [ "$FAIL_U" -ne 0 ] || [ "$FAIL_R" -ne 0 ]; then
        FAILED=1
    fi
elif [ "$DOMAIN_FILTER" = "understanding" ]; then
    train_one_domain "understanding" || FAILED=1
elif [ "$DOMAIN_FILTER" = "text_reformulation" ]; then
    train_one_domain "text_reformulation" || FAILED=1
else
    echo "Unknown domain: ${DOMAIN_FILTER}"
    echo "Usage: $0 [all|understanding|text_reformulation]"
    exit 1
fi

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
