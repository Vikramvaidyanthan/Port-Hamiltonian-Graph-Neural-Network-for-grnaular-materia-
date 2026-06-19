#!/bin/bash
#SBATCH --job-name=ph_gnn_train
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/work/venkatv1/ph_gnn_outputs/logs/ph_gnn_%j.out
#SBATCH --error=/scratch/work/venkatv1/ph_gnn_outputs/logs/ph_gnn_%j.err

# GPU
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100-80g,gpu-v100-32g


ROOT_DIR="/scratch/work/venkatv1"
SCRIPT="${ROOT_DIR}/ph_gnn_triton_v2.py"
OUT_DIR="${ROOT_DIR}/ph_gnn_outputs"

mkdir -p "${OUT_DIR}/logs"

echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "python=$(which python)"
echo "ROOT_DIR=${ROOT_DIR}"
echo "OUT_DIR=${OUT_DIR}"
date

module load triton/2025.1-gcc
module load mamba

source ~/.bashrc
source activate /scratch/work/venkatv1/envs/ph_gnn_env
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec python -u "${SCRIPT}" \
  --urdf "${ROOT_DIR}/Dataset/spot.urdf" \
  --wandb_project "ph-gnn-phase1" \
  --wandb_run "triton_a100_run1"

date
