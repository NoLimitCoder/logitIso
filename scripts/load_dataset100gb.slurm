#!/bin/bash
#SBATCH --job-name=dl-fineweb-64g
#SBATCH --time=23:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --output=../datasets/logs/dl_fineweb_100g.out
#SBATCH --error=../datasets/logs/dl_fineweb_100g.err

# Activate your venv
source /home/dionicio/projects/def-hsajjad/dionicio/logitIso/bin/activate
module load StdEnv/2023 python-build-bundle/2024a gcc arrow

export PYTHONUNBUFFERED=1

srun --unbuffered python -u ../datasets/prepare_fineweb.py

