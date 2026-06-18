#!/bin/bash
#SBATCH --job-name=ph_gnn_eval
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/work/venkatv1/ph_gnn_outputs/logs/ph_gnn_eval_%j.out
#SBATCH --error=/scratch/work/venkatv1/ph_gnn_outputs/logs/ph_gnn_eval_%j.err

ROOT_DIR="/scratch/work/venkatv1"
REPO_DIR="${ROOT_DIR}/ph_gnn"
SCRIPT="${REPO_DIR}/ph_gnn_eval_trainsplit.py"
OUT_DIR="${ROOT_DIR}/ph_gnn_outputs"
DATASET_ROOT="${ROOT_DIR}/Dataset"
MODEL_PATH="${OUT_DIR}/model_best.pt"
URDF_PATH="${DATASET_ROOT}/spot.urdf"
TRAIN_SPLIT_FRACTION="0.10"

mkdir -p "${OUT_DIR}/logs"

echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Script     : ${SCRIPT}"
echo "Model      : ${MODEL_PATH}"
echo "Dataset    : ${DATASET_ROOT} (first ${TRAIN_SPLIT_FRACTION} of training data)"
date

module load triton/2025.1-gcc
module load mamba

source /appl/scibuilder-mamba/aalto-rhel9/prod/software/mamba/2025.1/f67be15/etc/profile.d/conda.sh
conda activate /scratch/work/venkatv1/envs/ph_gnn_env

echo "Python     : $(which python)"
python --version

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "${SCRIPT}" ]; then
    echo "ERROR: script not found at ${SCRIPT}"
    exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: model_best.pt not found at ${MODEL_PATH}"
    exit 1
fi

if [ ! -d "${DATASET_ROOT}" ]; then
    echo "ERROR: dataset root not found at ${DATASET_ROOT}"
    exit 1
fi

python -u "${SCRIPT}" \
    --urdf                  "${URDF_PATH}" \
    --dataset_root          "${DATASET_ROOT}" \
    --test_subdir           "." \
    --model_path            "${MODEL_PATH}" \
    --out_dir               "${OUT_DIR}" \
    --train_split_fraction  "${TRAIN_SPLIT_FRACTION}" \
    --wandb_project         "ph-gnn-phase1" \
    --wandb_run             "trainsplit_eval_${SLURM_JOB_ID}"

echo "Done — $(date)"
