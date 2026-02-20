# Executed Notebook Guide: CNN DNA Motif Classifier

Source notebook: `cnn_dna_demo.ipynb`  
Executed artifact: `_executed_cnn.ipynb`

This guide explains the executed notebook in detail, including how sequence strings are encoded, how each custom function works, and how the CNN architecture maps to motif detection.

## 1) Problem Framing

Task: binary classification of DNA sequences into motif-positive vs motif-negative groups.

Class definition used in this notebook:

- Positive class (`1`): sequence has an inserted `TATA` motif.
- Negative class (`0`): random sequence without forced motif insertion.

Biological analogy:

- The model imitates promoter-motif detection from raw sequence without manual features.

## 2) Input Data Details

Dataset generation parameters:

- `n_pos=600`
- `n_neg=600`
- `SEQ_LEN=80`
- Alphabet: `A`, `T`, `C`, `G`
- Motif token: `TATA`

Raw data types:

- `seqs`: Python list of 1200 DNA strings.
- `labels`: NumPy int array of shape `(1200,)`.

Encoded tensor contract:

- One sequence -> one-hot array of shape `(80, 4)`.
- Dataset tensor stack -> `(N, 80, 4)`.
- After channel transpose for `Conv1d` -> `(N, 4, 80)`.

Important caveat for instruction:

- Random negatives can still contain `TATA` by chance.
- So the task has intentional label noise from motif collisions.

## 3) Method Summary

1. Generate synthetic DNA sequences with and without motif insertion.
2. Convert each sequence into one-hot numeric tensors.
3. Build a custom PyTorch `Dataset` and batched `DataLoader`.
4. Train a compact 1D CNN with binary cross-entropy on logits.
5. Evaluate at each epoch on held-out data.
6. Plot train/test loss curves and report final test accuracy.

## 4) Cell-by-Cell Walkthrough

### Code cell 1: Imports and seeds

- Imports NumPy, matplotlib, PyTorch modules.
- Seeds RNG with `RANDOM_STATE = 42`.
- Calls `torch.manual_seed(42)`.

### Code cell 2: Sequence synthesis functions and dataset build

Defines constants and helper functions, then executes `make_dataset()`.

Executed output shows:

- Positive example motif check.
- Negative example motif check (can be `True` due random chance).
- Total count and class rate (`1200`, `0.5`).

### Code cell 3: One-hot encoding and custom dataset class

- Defines base-to-index map.
- Defines `one_hot_encode(seq)`.
- Defines `DNADataset` with `__len__` and `__getitem__`.
- Returns tensors in `(channels, length)` format for convolution.

Executed sanity-check output:

- Example tensor shape: `torch.Size([4, 80])`
- Example label: `1.0`

### Code cell 4: Train/test split and loaders

- Uses first 80 percent for train and remaining 20 percent for test.
- Creates `DataLoader` objects with batch sizes 64 and 128.

Executed output:

- Train batches: `15`
- Test batches: `2`

### Code cell 5: CNN model definition

Defines `MotifCNN` architecture:

1. `Conv1d(4 -> 16, kernel_size=8)`
2. `ReLU`
3. `MaxPool1d(kernel_size=2)`
4. `Conv1d(16 -> 32, kernel_size=8)`
5. `ReLU`
6. `AdaptiveMaxPool1d(1)`
7. `Flatten`
8. `Linear(32 -> 1)`

Setup also includes:

- Device selection (`cuda` if available else `cpu`).
- `BCEWithLogitsLoss`.
- Adam optimizer with `lr=1e-3`.

### Code cell 6: Training and evaluation loop

- Defines `evaluate(loader)` to compute held-out loss and accuracy.
- Trains for `EPOCHS = 5`.
- For each batch: zero grad, forward, loss, backward, optimizer step.
- Logs epoch-level train loss, test loss, and test accuracy.

Executed epoch trace ends with:

- `Epoch 5/5 ... test_acc=0.671`

### Code cell 7: Loss plot and final metric

- Plots train and test loss trajectories.
- Prints final test accuracy.

Executed final metric:

- `Final test accuracy: 0.671`

## 5) Custom Function and Class Reference

### `random_dna(n, length=SEQ_LEN)`

- Input: number of sequences and length.
- Output: list of random DNA strings.
- Internal logic: samples from `ALPHABET` using seeded RNG.

### `insert_motif(seq, motif=MOTIF)`

- Input: one DNA sequence.
- Output: same-length sequence with motif injected at random position.
- Purpose: guarantees positive-class motif presence.

### `make_dataset(n_pos=600, n_neg=600)`

- Input: class sizes.
- Output: shuffled sequence list and aligned label array.
- Steps: create positives/negatives, concatenate, permute order.

### `one_hot_encode(seq)`

- Input: DNA string.
- Output: float32 matrix `(L, 4)`.
- Encoding: one active index per base from `BASE_TO_IDX` map.

### `DNADataset`

- Purpose: adapts sequence arrays to PyTorch dataset API.
- `__getitem__` returns:
- `x`: tensor `(4, L)` for `Conv1d`
- `y`: float scalar for BCE loss

### `MotifCNN`

- Purpose: learn motif-like filters and classify sequence.
- Returns one logit per sequence (`shape: (batch,)`).
- Uses global max pooling to capture strongest motif activation independent of absolute position.

### `evaluate(loader)`

- Purpose: inference-only metric collection.
- Uses `model.eval()` and `torch.no_grad()`.
- Returns average loss and accuracy.

## 6) Key PyTorch Function Explanations

| Function | What it does in this notebook |
|---|---|
| `nn.Conv1d` | Learns sequence-local filters analogous to motif detectors |
| `nn.MaxPool1d` | Reduces spatial length and keeps strong activations |
| `nn.AdaptiveMaxPool1d(1)` | Compresses variable activation map into one strongest value per channel |
| `nn.BCEWithLogitsLoss` | Stable binary loss on raw logits |
| `torch.sigmoid(logits)` | Converts logits to probabilities for thresholding |

## 7) How to Map This to Real Genomics Data

1. Read FASTA sequences from `data/`.
2. Standardize sequence length handling policy.
3. Add reverse-complement treatment if biologically relevant.
4. Split data by chromosome or experiment to avoid leakage.
5. Add AUROC and AUPRC for robust class-imbalance evaluation.

## 8) Key Teaching Message

This notebook shows the full raw-sequence workflow: symbolic biology data -> numeric tensor -> convolutional model -> interpretable motif-learning behavior.
