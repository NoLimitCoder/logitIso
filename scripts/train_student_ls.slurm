#!/bin/bash
#SBATCH --job-name=student-ls-135m
#SBATCH --time=1-23:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --output=../models/students/logs/train_student_ls.out
#SBATCH --error=../models/students/logs/train_student_ls.err

# Activate your env
source /home/dionicio/projects/def-hsajjad/dionicio/logitIso/bin/activate
module load StdEnv/2023 python-build-bundle/2024a gcc arrow

BASE=/home/dionicio/projects/def-hsajjad/dionicio/logitIso
DATA=/gpfs/project/def-hsajjad/dionicio/logitIso/datasets/fineweb_edu_dedup

srun --unbuffered python -u ${BASE}/models/students/ls_trainer.py \
  --train_dir ${DATA}/train_100gb \
  --eval_dir  ${DATA}/test_100gb \
  --output_dir ${BASE}/models/students/smolLM135MLabelSmoothing \
  --model_name HuggingFaceTB/SmolLM2-135M \
  --block_size 2048 \
  --train_bs 2 --eval_bs 2 \
  --grad_accum 16 \
  --epochs 1.0 \
  --lr 3e-4 \
  --ls 0.1 \
  --num_proc 32 \
  --save_steps 2000 \
  --eval_steps 2000 \
  --log_steps 50
