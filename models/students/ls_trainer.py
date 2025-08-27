#!/usr/bin/env python
"""
Train a SmolLM 135M student with label smoothing on a saved dataset.

Assumes you have:
  /gpfs/project/def-hsajjad/<user>/logitIso/datasets/fineweb_edu_dedup/train_100gb
  /gpfs/project/def-hsajjad/<user>/logitIso/datasets/fineweb_edu_dedup/test_100gb

Usage (example):
  python -u students/ls_trainer.py \
    --train_dir /gpfs/project/def-hsajjad/<user>/logitIso/datasets/fineweb_edu_dedup/train_100gb \
    --eval_dir  /gpfs/project/def-hsajjad/<user>/logitIso/datasets/fineweb_edu_dedup/test_100gb \
    --output_dir students/smolLM135MLabelSmoothing
"""

import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# -------- utils --------

def guess_text_column(ds) -> str:
    # Prefer common names
    for name in ["text", "content", "document", "raw"]:
        if name in ds.features and str(ds.features[name].dtype) in {"string", "large_string"}:
            return name
    # fallback: first string-ish column
    for k, v in ds.features.items():
        if "string" in str(v.dtype):
            return k
    raise ValueError("Could not find a text-like column in the dataset")

def get_block_size(tokenizer, requested: int) -> int:
    # Respect tokenizer limit if set
    lm = getattr(tokenizer, "model_max_length", None)
    if lm is None or lm > 100_000:  # “very large” sentinel values
        return requested
    return min(requested, lm)

def map_tokenize(ds, text_col, tokenizer, num_proc: int = 8):
    def tok(batch):
        # keep defaults (no padding, no truncation); we’ll pack later
        return tokenizer(batch[text_col])
    # Remove ALL original columns so only tokenized fields remain
    return ds.map(tok, batched=True, num_proc=num_proc, remove_columns=ds.column_names)

def group_texts(ds, block_size: int, num_proc: int = 8):
    # Pack contiguous tokens into fixed-size chunks
    def _group(examples: Dict[str, List[List[int]]]):
        # Keep only list-of-lists columns (e.g., input_ids, attention_mask)
        cols = {
            k: v for k, v in examples.items()
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list)
        }
        # Concatenate/flatten each column
        concatenated = {k: [tok for seq in v for tok in seq] for k, v in cols.items()}

        # Truncate to a multiple of block_size
        total_len = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_len, block_size)]
            for k, t in concatenated.items()
        }
        # Labels = inputs for causal LM
        result["labels"] = result["input_ids"].copy()
        return result

    # Drop pre-existing columns; keep only what _group returns
    return ds.map(_group, batched=True, num_proc=num_proc, remove_columns=ds.column_names)

# -------- main --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="Path to load_from_disk(train)")
    ap.add_argument("--eval_dir", required=True, help="Path to load_from_disk(eval)")
    ap.add_argument("--output_dir", default="students/smolLM135MLabelSmoothing")
    ap.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-135M",
                    help="Change if your 135M ID differs.")
    ap.add_argument("--block_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--train_bs", type=int, default=2, help="per-device train batch size")
    ap.add_argument("--eval_bs", type=int, default=2, help="per-device eval batch size")
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--ls", type=float, default=0.1, help="label smoothing factor")
    ap.add_argument("--num_proc", type=int, default=8)
    ap.add_argument("--save_steps", type=int, default=2000)
    ap.add_argument("--eval_steps", type=int, default=2000)
    ap.add_argument("--log_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets you saved earlier
    raw_train = load_from_disk(args.train_dir)
    raw_eval  = load_from_disk(args.eval_dir)

    text_col = guess_text_column(raw_train)

    # Tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    # pad_token may be missing for some causal LMs
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    block_size = get_block_size(tokenizer, args.block_size)

    # Tokenize + pack
    print(f"[ls_trainer] Tokenizing (num_proc={args.num_proc})…")
    tok_train = map_tokenize(raw_train, text_col, tokenizer, num_proc=args.num_proc)
    tok_eval  = map_tokenize(raw_eval,  text_col, tokenizer, num_proc=args.num_proc)

    print(f"[ls_trainer] Grouping into blocks of {block_size}…")
    train_ds = group_texts(tok_train, block_size, num_proc=args.num_proc)
    eval_ds  = group_texts(tok_eval,  block_size, num_proc=args.num_proc)

    # Model
    config = AutoConfig.from_pretrained(args.model_name)
    model  = AutoModelForCausalLM.from_pretrained(args.model_name, config=config)

    # Mixed precision flags
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    fp16 = torch.cuda.is_available() and not bf16

    # If using gradient checkpointing, disable cache to avoid warnings
    model.config.use_cache = False

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args (label smoothing enabled)
    targs = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        report_to=[],  # avoid wandb by default
        seed=args.seed,
        bf16=bf16,
        fp16=fp16,
        tf32=True,
        gradient_checkpointing=True,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.log_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="perplexity",
        greater_is_better=False,
        dataloader_num_workers=min(args.num_proc, 32),
        dataloader_pin_memory=True,
        label_smoothing_factor=args.ls,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,  # loss auto-logged; we derive PPL post-eval below
    )

    # Train
    train_result = trainer.train()
    trainer.save_model(args.output_dir)  # final weights + tokenizer + config
    tokenizer.save_pretrained(args.output_dir)

    # Final eval -> log perplexity
    eval_metrics = trainer.evaluate()
    try:
        eval_ppl = math.exp(eval_metrics["eval_loss"])
    except Exception:
        eval_ppl = float("inf")
    eval_metrics["perplexity"] = eval_ppl
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print("[ls_trainer] Done. Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
