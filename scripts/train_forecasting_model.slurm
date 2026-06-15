#!/bin/bash
#SBATCH --job-name=solar_forecast
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/train_%j_out.txt
#SBATCH --error=outputs/logs/train_%j_err.txt

echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Started      : $(date)"
echo "============================================"

# ── Load Python module ────────────────────────────────────────────────────────
module load python3

# ── Activate virtual environment ──────────────────────────────────────────────
source ~/envs/ai-energy-workshop/bin/activate

# ── Move to project directory ─────────────────────────────────────────────────
cd ~/ai-energy-workshop

# ── Create log directory ──────────────────────────────────────────────────────
mkdir -p outputs/logs

# ── Verify GPU ───────────────────────────────────────────────────────────────
echo ""
echo "GPU check:"
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# ── Run training ──────────────────────────────────────────────────────────────
echo ""
echo "Starting training..."
python3 scripts/train_forecasting_model.py \
    --epochs 300 \
    --patience 20 \
    --hidden 64 \
    --seed 469

echo ""
echo "============================================"
echo "Finished : $(date)"
echo "============================================"
