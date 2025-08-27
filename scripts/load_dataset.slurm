#!/bin/bash
#SBATCH --job-name=dl-fineweb-64g
#SBATCH --time=23:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --output=../datasets/logs/dl_fineweb_64g.out
#SBATCH --error=../datasets/logs/dl_fineweb_64g.err

# Activate your venv
source /home/dionicio/projects/def-hsajjad/dionicio/logitIso/bin/activate
pip install -q -U "huggingface_hub[cli]" hf_transfer

# Faster, resilient downloads
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_ENABLE_EMERGENCY_RETRY=1
export HF_HUB_EMERGENCY_RETRY_WAIT_TIME=15

mkdir -p ../datasets/fineweb_edu_dedup/parquet

hf download HuggingFaceTB/smollm-corpus --repo-type dataset \
  --include "fineweb-edu-dedup/train-*.parquet" \
  --local-dir ../datasets/fineweb_edu_dedup/parquet