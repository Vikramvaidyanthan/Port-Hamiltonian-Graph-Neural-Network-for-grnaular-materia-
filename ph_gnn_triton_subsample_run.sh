#!/bin/bash
#SBATCH --job-name=ph_gnn_subsample
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/work/venkatv1/ph_gnn_outputs/logs/ph_gnn_subsample_%j.out
#SBATCH --error=/scratch/work/venkatv1/ph_gnn_outputs/logs/ph_gnn_subsample_%j.err

# GPU
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100-80g

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR="/scratch/work/venkatv1"
SCRIPT="${ROOT_DIR}/ph_gnn/ph_gnn_triton_subsample.py"
OUT_DIR="${ROOT_DIR}/ph_gnn_outputs"

mkdir -p "${OUT_DIR}/logs"

# ── Diagnostics ──────────────────────────────────────────────────────────────
echo "========================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Job name     : ${SLURM_JOB_NAME}"
echo "Node         : $(hostname)"
echo "pwd          : $(pwd)"
echo "Partition    : ${SLURM_JOB_PARTITION}"
echo "CPUs         : ${SLURM_CPUS_PER_TASK}"
echo "Memory       : ${SLURM_MEM_PER_NODE} MB"
echo "GPU          : $(echo $CUDA_VISIBLE_DEVICES)"
echo "Script       : ${SCRIPT}"
echo "Out dir      : ${OUT_DIR}"
echo "Start time   : $(date)"
echo "========================================"

# ── Environment ──────────────────────────────────────────────────────────────
module load triton/2025.1-gcc
module load mamba

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/work/venkatv1/envs/ph_gnn_env

# ── Threading — prevent CPU over-subscription alongside GPU work ─────────────
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ── Python / CUDA settings ────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_${SLURM_JOB_ID}}"

# ── W&B offline fallback — set to "online" if cluster has internet access ─────
# export WANDB_MODE=offline

# ── GPU info ─────────────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
python -c "import torch; avail=torch.cuda.is_available(); print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda, '| device:', torch.cuda.get_device_name(0) if avail else 'CPU')"
echo "========================================"
echo "Launching training..."
echo "========================================"

# ── Training ─────────────────────────────────────────────────────────────────
exec python -u "${SCRIPT}" \
  --urdf      "${ROOT_DIR}/Dataset/spot.urdf" \
  --wandb_project "ph-gnn-phase1" \
  --wandb_run     "triton_subsample_run1"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
