#!/bin/bash
set -e

echo "=========================================="
echo "  API Testing Environment — Setup"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"
echo "Python version: $PYTHON_VERSION (required: >= $REQUIRED_VERSION)"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate
source .venv/bin/activate
echo "Activated virtual environment: $(which python)"

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install -r requirements.txt

# Login to HuggingFace (optional)
echo ""
echo "=========================================="
echo "  Optional: HuggingFace Login"
echo "=========================================="
echo "If you want to push models to HF Hub, run:"
echo "  huggingface-cli login"
echo ""

# Login to W&B (optional)
echo "=========================================="
echo "  Optional: Weights & Biases Login"
echo "=========================================="
echo "If you want W&B logging, run:"
echo "  wandb login"
echo ""

# Verify installation
echo "=========================================="
echo "  Verifying installation..."
echo "=========================================="
python -c "import fastapi; print(f'  fastapi: {fastapi.__version__}')"
python -c "import transformers; print(f'  transformers: {transformers.__version__}')"
python -c "import trl; print(f'  trl: {trl.__version__}')"
python -c "import peft; print(f'  peft: {peft.__version__}')"
python -c "import torch; print(f'  torch: {torch.__version__}')"
python -c "import wandb; print(f'  wandb: {wandb.__version__}')" 2>/dev/null || echo "  wandb: not installed (optional)"
python -c "import matplotlib; print(f'  matplotlib: {matplotlib.__version__}')" 2>/dev/null || echo "  matplotlib: not installed (optional)"
echo ""

echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "Quick commands:"
echo "  source .venv/bin/activate"
echo ""
echo "  # See training prompts (no GPU)"
echo "  SHOW_PROMPTS=1 python -m training.grpo"
echo ""
echo "  # Quick test (CPU, ~2 min)"
echo "  python -m training.grpo --test-mode"
echo ""
echo "  # Full training (GPU)"
echo "  python -m training.grpo --model-id Qwen/Qwen3-1.7B --num-episodes 100"
echo ""
echo "  # With HF push + W&B"
echo "  python -m training.grpo \\"
echo "    --push-to-hub --hf-repo-id your-name/api-tester-grpo \\"
echo "    --use-wandb --wandb-project api-testing-grpo"
echo ""
