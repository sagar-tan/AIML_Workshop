# Executed Notebook Guide: Protein Transformer Classifier

Source notebook: `transformer_protein_demo.ipynb`  
Executed artifact: `_executed_transformer.ipynb`

This guide documents the executed transformer notebook with detailed explanations of input encoding, custom classes, and training/evaluation flow.

## 1) Problem Framing

Task: classify protein sequences into two synthetic families using class-specific sequence patterns.

Class design:

- Class A (`0`): sequence includes pattern `CXXC`.
- Class B (`1`): sequence includes pattern `GGG`.

Teaching purpose:

- Demonstrate tokenization, positional encoding, and self-attention in a bio-sequence context.

## 2) Input Data Details

Synthetic generation settings:

- Number of Class A samples: `600`
- Number of Class B samples: `600`
- Sequence length: `120`
- Alphabet: 20 standard amino acids (`ACDEFGHIKLMNPQRSTVWY`)

Tokenization contract:

- Padding token: `_`
- Vocabulary: `[PAD] + 20 amino acids`
- Vocab size: `21`
- Each sequence converted to integer IDs of length `MAX_LEN = 120`
- Tensor shape for model input: `(batch, length)`

## 3) Method Summary

1. Generate random proteins and inject class-specific patterns.
2. Convert amino-acid strings to integer token IDs.
3. Build `Dataset`/`DataLoader` for batching.
4. Build transformer model with embedding + positional encoding + encoder stack.
5. Mean-pool sequence embeddings and classify with linear head.
6. Train with cross-entropy for 5 epochs.
7. Evaluate held-out loss/accuracy each epoch.
8. Plot train/test loss curves.

## 4) Cell-by-Cell Walkthrough

### Code cell 1: Imports and reproducibility

- Imports math, NumPy, matplotlib, PyTorch modules.
- Seeds NumPy and PyTorch random generators.

### Code cell 2: Synthetic sequence generation helpers

Defines:

- `random_protein`
- `insert_pattern`
- `make_dataset`

Then builds `seqs, labels = make_dataset()` and prints sanity checks.

Executed output confirms:

- Total sequences: `1200`
- Balanced classes (`Class B rate: 0.5`)

### Code cell 3: Tokenization and dataset class

- Creates vocabulary maps `stoi` and `itos`.
- Defines `tokenize(seq, max_len)`.
- Defines `ProteinDataset` returning `(token_ids, label)`.
- Runs sanity check.

Executed output confirms:

- Item token tensor shape: `torch.Size([120])`
- Vocab size: `21`

### Code cell 4: Train/test split and loaders

- 80/20 split without reshuffling after synthetic shuffle.
- Creates train/test dataloaders.

Executed output:

- Train batches: `15`
- Test batches: `2`

### Code cell 5: Transformer model construction

Defines classes:

- `PositionalEncoding`
- `ProteinTransformer`

Architecture details:

- `d_model=64`
- `nhead=4`
- `num_layers=2`
- Feed-forward width `128`
- Dropout `0.1`
- Mean pooling over sequence length dimension
- Linear output with `num_classes=2`

Also sets:

- Device selection.
- `CrossEntropyLoss`.
- Adam optimizer with learning rate `2e-3`.

### Code cell 6: Training/evaluation loop

- Defines `eval_loader(loader)` for loss and accuracy.
- Runs 5 training epochs.
- Logs train loss, test loss, and test accuracy each epoch.

Executed trend:

- Accuracy increases strongly by epoch 3 and stabilizes.
- Final epoch reports `test_acc=0.800`.

### Code cell 7: Plot and final metric

- Plots train/test cross-entropy curves.
- Prints final held-out accuracy.

Executed output:

- `Final test accuracy: 0.8`

## 5) Custom Function and Class Reference

### `random_protein(n, length=120)`

- Input: count and sequence length.
- Output: list of random protein strings.
- Use: generates class-neutral background sequences.

### `insert_pattern(seq, pattern_type)`

- Input: sequence and pattern selector (`CXXC` or `GGG`).
- Output: same-length sequence with class signature inserted.
- Notes:
- For `CXXC`, middle two residues are random amino acids.
- Raises `ValueError` for unknown pattern.

### `make_dataset(n_a=600, n_b=600, length=120)`

- Input: class sizes and length.
- Output: shuffled sequences and integer labels.
- Role: generates balanced supervised dataset.

### `tokenize(seq, max_len=MAX_LEN)`

- Input: amino-acid string.
- Output: int64 array of token IDs length `max_len`.
- Behavior:
- Truncates if too long.
- Right-pads with `PAD` token if too short.

### `ProteinDataset`

- Purpose: PyTorch dataset wrapper for tokenized sequences.
- Returns pair:
- `torch.from_numpy(self.X[idx])` tokens
- `torch.tensor(self.y[idx])` label

### `PositionalEncoding`

- Purpose: add deterministic position signals to embeddings.
- Implementation:
- Precomputes sinusoidal matrix `pe`.
- Stores with `register_buffer` so it moves with device but is not trainable.

### `ProteinTransformer`

- Purpose: sequence classifier using transformer encoder blocks.
- Forward path:
1. Embed token IDs.
2. Add positional encoding.
3. Apply encoder layers.
4. Mean-pool over sequence positions.
5. Apply linear classifier head.

### `eval_loader(loader)`

- Purpose: evaluate loss and accuracy without gradient computation.
- Returns average loss and proportion correct.

## 6) Core Transformer Function Explanations

| API | Notebook role |
|---|---|
| `nn.Embedding` | Learns dense vector per token ID |
| `nn.TransformerEncoderLayer` | Implements multi-head self-attention + feed-forward block |
| `nn.TransformerEncoder` | Stacks encoder blocks for richer contextualization |
| `CrossEntropyLoss` | Multi-class classification loss on logits |
| `logits.argmax(dim=1)` | Converts class scores to predicted labels |

## 7) Mapping to Real Protein ML Workflows

1. Replace synthetic generator with FASTA parsing and family labels.
2. Keep tokenization scaffold, optionally with special tokens.
3. Add masking-aware pooling for variable lengths.
4. Split by homologous clusters to avoid leakage.
5. Compare from-scratch model vs pretrained protein language models.

## 8) Key Teaching Message

Participants should leave understanding that transformers are not magic: they are a sequence of concrete tensor operations that can be inspected, modified, and validated for biological research tasks.
