# logitIso Experiments

This repo orchestrates the experiments for studying **logit scrambling** and its effect on model distillability.

## Level 0 (this commit)
- Cache baseline teachers locally:
- **Teacher_Large_Orig** → `HuggingFaceTB/SmolLM2-1.7B`
- **Teacher_Small_Control** → `HuggingFaceTB/SmolLM2-360M`
- Start student baseline training with **label smoothing only** (no KD).

## Quickstart

### 1) Install
```bash
conda create -n logitnoise python=3.10 -y
conda activate logitnoise
pip install -r requirements.txt