#!/bin/bash
# ============================================================
# API Testing Environment — One-command setup
# ============================================================
# Usage: bash setup.sh
#
# This script:
#   1. Creates a virtual environment
#   2. Detects your GPU and installs the correct PyTorch+CUDA
#   3. Installs all project dependencies
#   4. Verifies everything works
# ============================================================

set -e

echo ""
echo "============================================"
echo "  API Testing Environment — Setup"
echo "============================================"
echo ""

# --- Step 1: Create venv ---
echo "[1/5] Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  Created .venv"
else
    echo "  .venv already exists"
fi
source .venv/bin/activate
pip install --upgrade pip setuptools wheel -q
echo "  Python: $(python3 --version)"
echo "  pip: $(pip --version | awk '{print $2}')"
echo ""

# --- Step 2: Install PyTorch with correct CUDA ---
echo "[2/5] Detecting GPU and installing PyTorch..."

install_pytorch() {
    if command -v nvidia-smi &> /dev/null; then
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)

        echo "  GPU: $GPU_NAME ($GPU_MEM)"
        echo "  NVIDIA driver: $DRIVER_VERSION"

        if [ "$DRIVER_MAJOR" -ge 530 ]; then
            echo "  -> Installing PyTorch + CUDA 12.1"
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
        elif [ "$DRIVER_MAJOR" -ge 450 ]; then
            echo "  -> Installing PyTorch + CUDA 11.8 (older driver)"
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
        else
            echo "  WARNING: Driver too old ($DRIVER_VERSION). Install CPU PyTorch."
            echo "  Upgrade: https://www.nvidia.com/Download/index.aspx"
            pip install torch torchvision -q
        fi
    else
        echo "  No NVIDIA GPU detected."
        # Check for Apple Silicon
        if python3 -c "import platform; exit(0 if platform.processor() == 'arm' else 1)" 2>/dev/null; then
            echo "  -> Apple Silicon detected, installing default PyTorch (MPS support)"
        else
            echo "  -> Installing CPU-only PyTorch"
        fi
        pip install torch torchvision -q
    fi
}

install_pytorch
echo ""

# --- Step 3: Install project dependencies ---
echo "[3/5] Installing project dependencies..."
pip install -r requirements.txt -q
echo "  Done."
echo ""

# --- Step 4: Verify everything ---
echo "[4/5] Verifying installation..."
echo ""
python3 << 'PYEOF'
import sys

# Core
import fastapi, uvicorn, pydantic, httpx
print(f"  fastapi:      {fastapi.__version__}")

# ML
import torch
print(f"  torch:        {torch.__version__}")
cuda = torch.cuda.is_available()
mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
if cuda:
    print(f"  CUDA:         {torch.version.cuda}")
    print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory:   {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
elif mps:
    print(f"  Device:       Apple MPS")
else:
    print(f"  Device:       CPU only (training will be slow!)")

import transformers, trl, peft, datasets
print(f"  transformers: {transformers.__version__}")
print(f"  trl:          {trl.__version__}")
print(f"  peft:         {peft.__version__}")

# Optional
try:
    import wandb
    print(f"  wandb:        {wandb.__version__}")
except ImportError:
    print(f"  wandb:        not installed (optional)")

try:
    import gradio
    print(f"  gradio:       {gradio.__version__}")
except ImportError:
    print(f"  gradio:       not installed (optional)")

# OpenEnv
try:
    import openenv
    print(f"  openenv:      OK")
except ImportError:
    print(f"  openenv:      MISSING — run: pip install -r requirements.txt")

# Environment test
print("")
sys.path.insert(0, ".")
from server.environment import APITestEnvironment
from models import APITestAction, HTTPMethod
env = APITestEnvironment()
obs = env.reset(seed=42, task_id="basic_validation")
obs = env.step(APITestAction(method=HTTPMethod.GET, endpoint="/tasks/999999", expected_status=404))
assert obs.bugs_found_so_far == 1, "Bug detection failed!"
print(f"  Environment:  OK (bug detection verified)")
PYEOF

echo ""

# --- Step 5: Done ---
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Activate:     source .venv/bin/activate"
echo ""
echo "  Gradio UI:    python gradio_app.py"
echo "  Baselines:    python -m training.evaluate --task all --agent all"
echo "  Training:     python -m training.grpo --model-id Qwen/Qwen3-1.7B"
echo "  Test mode:    python -m training.grpo --test-mode"
echo ""
echo "  For HF Hub:   huggingface-cli login"
echo "  For W&B:      wandb login"
echo ""
