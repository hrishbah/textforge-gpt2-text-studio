"""Fine-tune GPT-2 on a custom text dataset, with a fast smoke-test mode.

Examples
--------
Smoke test (validates the whole pipeline in a few minutes):

    python train.py --model-name gpt2 --data-path data/train.txt \
        --output-dir models/gpt2-finetuned \
        --epochs 3 --batch-size 2 --block-size 128 --max-steps 500

Full fine-tuning (substantially longer):

    python train.py --model-name gpt2 --data-path data/train.txt \
        --output-dir models/gpt2-finetuned \
        --epochs 10 --batch-size 4 --block-size 256 --max-steps 2000

The trained model and tokenizer are always saved to ``--output-dir`` so the
Streamlit app can detect and load the fine-tuned checkpoint.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# Make the project root importable when this file is run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hfcache"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.textforge.config import (
    DEFAULT_MAX_TRAIN_STEPS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_TRAIN_BLOCK_SIZE,
    DEFAULT_TRAIN_EPOCHS,
    MIN_DATASET_CHARS,
)
from src.textforge.dataset import DatasetError, build_token_blocks, load_dataset
from src.textforge.model_manager import ModelError, get_device, load_model


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 on a custom text dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Base model id to start from (e.g. 'gpt2').",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the training .txt file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/gpt2-finetuned",
        help="Directory where the fine-tuned model and tokenizer are saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_TRAIN_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE,
        help="Per-device batch size. Keep small (1-4) on a laptop.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_TRAIN_BLOCK_SIZE,
        help="Sequence length in tokens per training example.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_TRAIN_STEPS,
        help="Maximum optimiser steps (0 = no limit).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Force a device ('cpu', 'mps', 'cuda'). Auto-detected when empty.",
    )
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_disk_space(output_dir: Path, required_mb: int = 1500) -> None:
    """Warn (and stop) when there is not enough free disk space."""
    try:
        usage = shutil.disk_usage(output_dir if output_dir.exists() else Path("."))
    except OSError:
        return
    free_mb = usage.free / (1024 * 1024)
    if free_mb < required_mb:
        raise SystemExit(
            f"Not enough free disk space: {free_mb:.0f} MB available but "
            f"~{required_mb} MB is needed for the model weights and checkpoints."
        )
    print(f"Free disk space: {free_mb:.0f} MB")


def main(argv: list[str] | None = None) -> int:
    """Entry point for the fine-tuning script."""
    args = parse_args(argv)
    set_seed(args.seed)

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)

    # Never start training in a directory that already contains files.
    # This protects saved checkpoints and training artifacts.
    if output_dir.exists() and any(output_dir.iterdir()):
        print(
            f"Refusing to train: output directory is not empty: {output_dir}\n"
            "Choose a new --output-dir to preserve existing files.",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------ #
    # 1. Validate the dataset BEFORE touching the model.
    # ------------------------------------------------------------------ #
    print(f"[1/6] Validating dataset: {data_path}")
    try:
        content, paragraphs = load_dataset(data_path)
    except DatasetError as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return 1

    if len(content.strip()) < MIN_DATASET_CHARS:
        print(
            f"Dataset error: only {len(content.strip())} characters, "
            f"minimum is {MIN_DATASET_CHARS}.",
            file=sys.stderr,
        )
        return 1
    print(f"      {len(paragraphs)} paragraph(s), {len(content)} characters")

    check_disk_space(output_dir)

    # ------------------------------------------------------------------ #
    # 2. Load the real pretrained model + tokenizer.
    # ------------------------------------------------------------------ #
    print(f"[2/6] Loading base model '{args.model_name}'...")
    try:
        model, tokenizer, resolved_id = load_model(args.model_name)
    except ModelError as exc:
        print(f"Model loading failed: {exc}", file=sys.stderr)
        return 1
    print(f"      loaded '{resolved_id}'")

    # GPT-2 has no pad token -> reuse EOS for padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ------------------------------------------------------------------ #
    # 3. Tokenise into fixed-size blocks (uniform batches).
    # ------------------------------------------------------------------ #
    print(f"[3/6] Tokenising into {args.block_size}-token blocks...")
    try:
        examples = build_token_blocks(content, tokenizer, block_size=args.block_size)
    except (DatasetError, ValueError) as exc:
        print(f"Tokenisation error: {exc}", file=sys.stderr)
        return 1
    print(f"      {len(examples)} training blocks")

    # ------------------------------------------------------------------ #
    # 4. Device + mixed precision selection.
    # ------------------------------------------------------------------ #
    device = args.device or get_device()
    print(f"[4/6] Device: {device}")

    use_bf16 = False
    use_fp16 = False
    if device == "cuda":
        use_fp16 = torch.cuda.is_available()
        use_bf16 = bool(torch.cuda.is_bf16_supported())
    elif device == "cpu":
        # CPU bf16 support varies; keep it off for maximum compatibility.
        use_bf16 = False

    # ------------------------------------------------------------------ #
    # 5. Configure the trainer.
    # ------------------------------------------------------------------ #
    print("[5/6] Configuring trainer...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        use_cpu=(device == "cpu"),
        use_mps_device=(device == "mps"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=min(100, max(1, args.max_steps // 10 or 1)),
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        fp16=use_fp16,
        bf16=use_bf16,
        optim="adamw_torch",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to=[],
        disable_tqdm=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=examples,
        data_collator=data_collator,
    )

    # ------------------------------------------------------------------ #
    # 6. Train and save.
    # ------------------------------------------------------------------ #
    print("[6/6] Starting training...\n")
    start = time.time()
    try:
        result = trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by the user. Nothing was saved.", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface any trainer failure clearly
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    elapsed = time.time() - start

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = getattr(result, "metrics", {}) or {}
    train_loss = metrics.get("train_loss")

    print("\n" + "=" * 62)
    print("Training completed")
    print("=" * 62)
    print(f"  Duration      : {elapsed / 60:.1f} minutes")
    print(f"  Steps         : {metrics.get('train_steps', 'n/a')}")
    if train_loss is not None:
        print(f"  Final loss    : {float(train_loss):.4f}")
    print(f"  Checkpoint    : {output_dir.resolve()}")
    print("=" * 62)
    print("\nRestart the Streamlit app and select 'Fine-tuned GPT-2 (local)':")
    print("    streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
