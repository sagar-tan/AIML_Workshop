# DNA Motif Detection with 1D CNN

This project teaches how sequence-first deep learning works for regulatory genomics style tasks.

## Learning Objectives

- Convert raw DNA strings into model-ready tensors.
- Understand 1D convolution as a motif scanning operation.
- Train a binary classifier with `BCEWithLogitsLoss`.
- Evaluate training dynamics with loss curves and held-out accuracy.

## Files

- `cnn_dna_demo.ipynb`: source notebook.
- `_executed_cnn.ipynb`: executed run with outputs.
- `_executed_cnn.md`: detailed function/class walkthrough for participant instruction.
- `data/`: placeholder for real FASTA-based promoter datasets.

## Input Data (Current Demo)

The notebook generates synthetic DNA to keep runtime short and setup simple.

- Total sequences: `1200`
- Positive class: `600` sequences with inserted motif `TATA`
- Negative class: `600` random sequences
- Sequence length: `80`
- Alphabet: `A`, `T`, `C`, `G`

Tensor representation:

- Before PyTorch conversion: `(N, L, 4)` one-hot arrays.
- During model training: `(N, 4, L)` to match `Conv1d` channel-first format.

Important teaching caveat:

- Random negatives may still contain `TATA` by chance because motif length is short.
- This makes the task realistically noisy and helps explain imperfect test accuracy.

## Method (Notebook Execution Flow)

1. Set reproducible random seeds for NumPy and PyTorch.
2. Generate synthetic positive and negative DNA sequences.
3. One-hot encode each sequence base into length-by-4 matrices.
4. Wrap data in a custom `Dataset` and `DataLoader`.
5. Build a small 1D CNN: conv -> relu -> pool -> conv -> relu -> global pool -> dense.
6. Train with Adam and `BCEWithLogitsLoss` for 5 epochs.
7. Evaluate test loss and test accuracy after each epoch.
8. Plot train/test loss curves.

## Function and Class Reference

### Custom functions

| Name | Input | Output | Role |
|---|---|---|---|
| `random_dna(n, length)` | number of sequences, length | list of DNA strings | Generates random background DNA |
| `insert_motif(seq, motif)` | one sequence | modified sequence | Injects known motif into positive examples |
| `make_dataset(n_pos, n_neg)` | class counts | `seqs`, `labels` | Creates shuffled supervised dataset |
| `one_hot_encode(seq)` | DNA string | `(L, 4)` float matrix | Converts categorical bases to numeric tensor |
| `evaluate(loader)` | DataLoader | `(loss, accuracy)` | Runs model in eval mode without gradients |

### Custom classes

| Class | Purpose | Key details |
|---|---|---|
| `DNADataset` | PyTorch dataset wrapper | Stores one-hot arrays and returns `(4, L)` tensors plus float labels |
| `MotifCNN` | Binary sequence classifier | Two conv blocks and adaptive pooling before linear output logit |

### Core PyTorch calls worth teaching

| Call | Why it matters |
|---|---|
| `nn.Conv1d(in_channels=4, ...)` | Learns position-local motif detectors over sequence |
| `nn.AdaptiveMaxPool1d(1)` | Aggregates strongest motif evidence regardless of position |
| `nn.BCEWithLogitsLoss()` | Stable binary loss combining sigmoid + BCE |
| `torch.sigmoid(logits) > 0.5` | Converts logits into hard class predictions |

## Last Executed Results

From `_executed_cnn.ipynb`:

- Train batches: `15`
- Test batches: `2`
- Final test accuracy: `0.671`
- Observed learning trend: gradual reduction in loss with moderate generalization.

## Real-Data Adaptation Path

1. Load promoter/non-promoter FASTA files from `data/`.
2. Add consistent sequence length policy: trim, pad, or sliding windows.
3. Address class imbalance with weighted loss or sampling.
4. Add validation split and early stopping.
5. Add saliency or filter visualization for motif interpretability.

## Teaching Notes for Participants

- Explain that CNN filters are trainable motif scanners.
- Show how tensor shape changes from string to matrix to channels-first tensors.
- Highlight the distinction between train mode and eval mode.
- Connect motif detection to transcription-factor binding intuition.

## Suggested Next Iterations

1. Increase channel width and compare performance.
2. Introduce multi-motif synthetic labels.
3. Add reverse-complement augmentation.
4. Compare CNN against transformer on the same DNA task.
