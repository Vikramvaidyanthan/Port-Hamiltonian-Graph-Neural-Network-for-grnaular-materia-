#!/bin/bash
#SBATCH --job-name=phgnn_eval_subsample
#SBATCH --output=/scratch/work/venkatv1/logs/eval_subsample_%j.out
#SBATCH --error=/scratch/work/venkatv1/logs/eval_subsample_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=venkatv1@aalto.fi

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load anaconda cuda/12.1

conda activate phgnn          # adjust to your actual conda env name

# Ensure log directory exists
mkdir -p /scratch/work/venkatv1/logs

# ── W&B: cache dir on scratch (home quota is small on Triton) ────────────────
export WANDB_DIR=/scratch/work/venkatv1/wandb_cache
mkdir -p "$WANDB_DIR"

# Optional: run offline and sync later if the compute node has no internet
# export WANDB_MODE=offline
# After the job: wandb sync /scratch/work/venkatv1/wandb_cache/wandb/offline-run-*

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT=/home/rllab/msc_student/Vikram/scripts/ph_gnn_eval_zigzag_subsample.py
URDF=/home/rllab/msc_student/Vikram/assets/spot.urdf
DATASET_ROOT=/scratch/work/venkatv1/Dataset
MODELS_DIR=/scratch/work/venkatv1/ph_gnn_outputs
OUT_DIR=/scratch/work/venkatv1/ph_gnn_outputs

# ── Job info ──────────────────────────────────────────────────────────────────
echo "======================================================"
echo " Job ID        : $SLURM_JOB_ID"
echo " Node          : $SLURMD_NODENAME"
echo " GPUs          : $CUDA_VISIBLE_DEVICES"
echo " Start time    : $(date)"
echo " Models dir    : $MODELS_DIR"
echo " Script        : $SCRIPT"
echo "======================================================"

srun python "$SCRIPT" \
    --urdf          "$URDF"                        \
    --dataset_root  "$DATASET_ROOT"                \
    --test_subdir   DatasetTest_zigzag             \
    --models_dir    "$MODELS_DIR"                  \
    --out_dir       "$OUT_DIR"                     \
    --wandb_project ph-gnn-phase1                  \
    --wandb_run     zigzag_eval_subsample_epochs

EXIT_CODE=$?
echo "======================================================"
echo " End time   : $(date)"
echo " Exit code  : $EXIT_CODE"
echo "======================================================"
exit $EXIT_CODE
