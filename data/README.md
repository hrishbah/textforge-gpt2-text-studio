# Training Dataset

This directory contains `train.txt` — a small demonstration corpus for fine-tuning GPT-2.

## About this dataset

This is an **original, small-scale** text corpus (~2KB, ~30 lines). It is NOT intended for production use but serves as:

- A quick smoke test for the training pipeline
- A template for users who want to plug in their own data
- An example of how to format a text training file

## Customizing the dataset

Replace `train.txt` with your own text corpus:

1. Create a new `.txt` file with UTF-8 encoding
2. Write one paragraph per block, separated by blank lines
3. Ensure the file is at least ~500 characters with ~5+ non-empty lines
4. Run `python train.py --data-path data/train.txt`

## Technical notes

- Each line is stripped of leading/trailing whitespace
- Paragraphs are joined for chunking into fixed-size training blocks
- The dataset is tokenised on-the-fly during training
