#!/usr/bin/env bash
# Create / update the domain_sd conda environment and install GPU packages.
#
# Usage:
#   bash setup_server.sh          # full install (includes vLLM + DeepSpeed)
#   bash setup_server.sh --no-gpu # skip vLLM + DeepSpeed

set -euo pipefail

ENV_NAME="domain_sd"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_GPU=true

for arg in "$@"; do
  [[ "$arg" == "--no-gpu" ]] && INSTALL_GPU=false
done

log()  { echo -e "\n\033[1;34m▶ $*\033[0m"; }
ok()   { echo -e "\033[1;32m✔ $*\033[0m"; }
warn() { echo -e "\033[1;33m⚠ $*\033[0m"; }

# ── 1. Conda environment ──────────────────────────────────────────────────────
log "Setting up conda environment '$ENV_NAME'"
if conda env list | grep -q "^${ENV_NAME} "; then
    warn "Environment '$ENV_NAME' already exists — updating from environment.yml"
    conda env update -n "$ENV_NAME" \
        --file "$SCRIPT_DIR/environment.yml" \
        --prune
else
    conda env create -f "$SCRIPT_DIR/environment.yml"
fi
ok "Conda environment '$ENV_NAME' ready"

# ── 2. GPU packages (vLLM + DeepSpeed) ────────────────────────────────────────
if $INSTALL_GPU; then
    log "Installing GPU packages (vLLM, DeepSpeed)"
    if command -v nvcc &>/dev/null; then
        ok "CUDA $(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')" detected
    else
        warn "nvcc not found — vLLM will install a bundled CUDA runtime"
    fi
    conda run -n "$ENV_NAME" pip install --upgrade \
        "vllm>=0.4.0" \
        "deepspeed>=0.14.0"
    ok "vLLM + DeepSpeed installed"

    log "Patching vLLM registry for BambooForCausalLM (TurboSparse-Mistral)"
    conda run -n "$ENV_NAME" python "$SCRIPT_DIR/patch_vllm.py"
    ok "vLLM registry patched"
else
    warn "Skipping GPU packages (--no-gpu)"
fi

# ── 3. Credentials ────────────────────────────────────────────────────────────
log "Checking credentials"
if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    warn ".env created from .env.example — fill in S3_SECRET_KEY before running scripts"
else
    ok ".env already present"
fi

# ── 4. Smoke test ─────────────────────────────────────────────────────────────
log "Smoke test"
conda run -n "$ENV_NAME" python - <<'PYEOF'
import torch, transformers, hydra, mlflow, boto3
print(f"  torch        {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
print(f"  transformers {transformers.__version__}")
print(f"  hydra        {hydra.__version__}")
print(f"  mlflow       {mlflow.__version__}")
print(f"  boto3        {boto3.__version__}")
if torch.cuda.is_available():
    print(f"  GPU          {torch.cuda.get_device_name(0)}")
PYEOF

if $INSTALL_GPU; then
    conda run -n "$ENV_NAME" python - <<'PYEOF'
try:
    import vllm, deepspeed
    print(f"  vllm         {vllm.__version__}")
    print(f"  deepspeed    {deepspeed.__version__}")
except ImportError as e:
    print(f"  WARNING: {e}")
PYEOF
fi
ok "Smoke test passed"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete.  conda activate $ENV_NAME"
echo ""
echo "  Download models + data:  python src/download_from_s3.py --all"
echo "  Generate data:           python src/generate_synthetic_data.py"
echo "  MLflow UI:               mlflow ui --backend-store-uri ./mlruns"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
