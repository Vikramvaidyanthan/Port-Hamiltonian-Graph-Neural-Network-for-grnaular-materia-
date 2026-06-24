#!/bin/bash
#SBATCH --job-name=ph_gnn_eval
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/work/venkatv1/ph_gnn_outputs/logs/ph_gnn_eval_%j.out
#SBATCH --error=/scratch/work/venkatv1/ph_gnn_outputs/logs/ph_gnn_eval_%j.err

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR="/scratch/work/venkatv1"
SCRIPT="${ROOT_DIR}/ph_gnn_eval_zigzag.py"
OUT_DIR="${ROOT_DIR}/ph_gnn_outputs"
ENV_DIR="${ROOT_DIR}/envs/ph_gnn_env"

DATASET_ROOT="${ROOT_DIR}/Dataset"
TEST_SUBDIR="DatasetTest_zigzag"
MODEL_PATH="${OUT_DIR}/model_best.pt"
URDF_PATH="${DATASET_ROOT}/spot.urdf"

# ── Environment ────────────────────────────────────────────────────────────────
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
date

source "${ENV_DIR}/bin/activate"
python="${ENV_DIR}/bin/python"

echo "Python     : ${python}"
echo "Script     : ${SCRIPT}"
echo "Model      : ${MODEL_PATH}"
echo "Test data  : ${DATASET_ROOT}/${TEST_SUBDIR}"

# ── Sanity checks ──────────────────────────────────────────────────────────────
if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: model_best.pt not found at ${MODEL_PATH}"
    exit 1
fi

if [ ! -d "${DATASET_ROOT}/${TEST_SUBDIR}" ]; then
    echo "ERROR: Test dataset not found at ${DATASET_ROOT}/${TEST_SUBDIR}"
    exit 1
fi

mkdir -p "${OUT_DIR}/logs"

# ── Run evaluation ─────────────────────────────────────────────────────────────
${python} "${SCRIPT}" \
    --urdf          "${URDF_PATH}" \
    --dataset_root  "${DATASET_ROOT}" \
    --test_subdir   "${TEST_SUBDIR}" \
    --model_path    "${MODEL_PATH}" \
    --out_dir       "${OUT_DIR}" \
    --wandb_project "ph-gnn-phase1" \
    --wandb_run     "zigzag_eval_${SLURM_JOB_ID}"

echo "Done — $(date)"
