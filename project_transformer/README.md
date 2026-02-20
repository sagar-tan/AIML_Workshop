# Protein Family Classification with Transformer Encoder

This project teaches sequence tokenization, positional encoding, and self-attention for protein sequence classification.

## Learning Objectives

- Convert amino-acid strings into fixed-length token tensors.
- Understand why positional encoding is required in transformer architectures.
- Train a compact transformer encoder classifier.
- Evaluate sequence classification with cross-entropy and accuracy.

## Files

- `transformer_protein_demo.ipynb`: source notebook.
- `_executed_transformer.ipynb`: executed run with outputs.
- `_executed_transformer.md`: detailed function/class and cell-by-cell guide.
- `data/`: location for real protein sequences and labels.

## Input Data (Current Demo)

The notebook creates synthetic protein sequences for reproducibility and speed.

- Total sequences: `1200`
- Class A (`0`): contains an inserted `CXXC` pattern
- Class B (`1`): contains inserted `GGG`
- Sequence length: `120`
- Alphabet: 20 standard amino acids

Tokenization setup:

- Vocabulary size: `21` (`20` amino acids + `PAD` token `_`).
- Sequence tensor shape after tokenization: `(N, 120)` integer IDs.

## Method (Notebook Execution Flow)

1. Set deterministic seeds for NumPy and PyTorch.
2. Generate random protein sequences and inject class-specific patterns.
3. Tokenize amino acids to integer IDs and pad/truncate to fixed length.
4. Build train/test datasets and dataloaders.
5. Construct transformer model:
- embedding layer
- sinusoidal positional encoding
- stacked encoder blocks
- mean pooling and linear classification head
6. Train for 5 epochs using Adam and cross-entropy.
7. Evaluate held-out loss and accuracy each epoch.
8. Plot train/test loss curves.

## Function and Class Reference

### Custom functions

| Name | Input | Output | Role |
|---|---|---|---|
| `random_protein(n, length)` | sample count, length | list of sequence strings | Creates random amino-acid backgrounds |
| `insert_pattern(seq, pattern_type)` | sequence, class pattern | edited sequence | Injects `CXXC` or `GGG` signatures |
| `make_dataset(n_a, n_b, length)` | class sizes, length | `seqs`, `labels` | Builds shuffled labeled dataset |
| `tokenize(seq, max_len)` | sequence string | integer array | Converts sequence to fixed-size token IDs |
| `eval_loader(loader)` | DataLoader | `(loss, accuracy)` | Evaluation loop with no gradient updates |

### Custom classes

| Class | Purpose | Key details |
|---|---|---|
| `ProteinDataset` | Provides token IDs and class labels | Returns `(L,)` token tensor and scalar label |
| `PositionalEncoding` | Injects absolute position information | Uses sinusoidal encoding registered as non-trainable buffer |
| `ProteinTransformer` | End-to-end sequence classifier | Embedding + positional encoding + transformer encoder + mean pooling + linear head |

### Core transformer concepts shown directly in code

| Concept | Implementation in notebook |
|---|---|
| Token embedding | `nn.Embedding(vocab_size, d_model, padding_idx=...)` |
| Positional information | `PositionalEncoding.forward` adds precomputed sine/cosine matrix |
| Self-attention stack | `nn.TransformerEncoder` over `nn.TransformerEncoderLayer` |
| Sequence aggregation | `x.mean(dim=1)` mean-pools token representations |
| Class prediction | `nn.Linear(d_model, num_classes)` head |

## Last Executed Results

From `_executed_transformer.ipynb`:

- Train batches: `15`
- Test batches: `2`
- Final test accuracy: `0.800`
- Training trend: clear improvement after early epochs, then stabilization.

## Real-Data Adaptation Path

1. Replace synthetic generator with FASTA parsing + family labels.
2. Keep explicit train/validation/test split by protein family.
3. Add masking for variable sequence lengths.
4. Add class weighting for imbalanced families.
5. Compare this small model to pretrained protein language models.

## Teaching Notes for Participants

- Emphasize why self-attention handles nonlocal dependencies better than fixed kernels.
- Explain that positional encoding is essential because attention itself is permutation-invariant.
- Show how sequence-to-label pipelines in proteomics mirror NLP pipelines.
- Discuss trade-offs between interpretability, compute cost, and performance.

## Suggested Next Iterations

1. Increase number of encoder layers and heads.
2. Add attention map visualization for biological interpretation.
3. Fine-tune ESM or ProtBERT embeddings instead of training from scratch.
4. Extend from binary family classification to multi-label functional annotation.
