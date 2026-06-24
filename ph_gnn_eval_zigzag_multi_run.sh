#!/bin/bash
#SBATCH --job-name=ph_gnn_eval
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00          # increased — 20 models × ~2min each = ~40min minimum
#SBATCH --output=/scratch/work/venkatv1/ph_gnn_outputs/logs/ph_gnn_eval_%j.out
#SBATCH --error=/scratch/work/venkatv1/ph_gnn_outputs/logs/ph_gnn_eval_%j.err

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR="/scratch/work/venkatv1"
SCRIPT="${ROOT_DIR}/ph_gnn/ph_gnn_eval_zigzag_multi.py"
OUT_DIR="${ROOT_DIR}/ph_gnn_outputs"
ENV_DIR="${ROOT_DIR}/envs/ph_gnn_env"

DATASET_ROOT="${ROOT_DIR}/Dataset"
TEST_SUBDIR="DatasetTest_zigzag"
MODELS_DIR="${OUT_DIR}"             # folder containing model_best_1.pt ... model_best_20.pt
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
echo "Models dir : ${MODELS_DIR}"
echo "Test data  : ${DATASET_ROOT}/${TEST_SUBDIR}"

# ── Sanity checks ──────────────────────────────────────────────────────────────
# Check at least one model_best_N.pt exists
if ! ls "${MODELS_DIR}"/model_best_*.pt 1>/dev/null 2>&1; then
    echo "ERROR: No model_best_*.pt files found in ${MODELS_DIR}"
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
    --models_dir    "${MODELS_DIR}" \
    --out_dir       "${OUT_DIR}" \
    --wandb_project "ph-gnn-phase1" \
    --wandb_run     "zigzag_eval_${SLURM_JOB_ID}"

echo "Done — $(date)"
