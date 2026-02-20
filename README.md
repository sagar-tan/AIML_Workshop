# Bioinformatics AI Teaching Workshop

This repository contains four compact, reproducible machine-learning projects designed for participants learning AI for bioinformatics.

All projects follow the same core workflow:

1. Define a biological prediction task.
2. Convert biological records to tensors.
3. Train a model family suited to the data modality.
4. Evaluate with clear metrics.
5. Interpret model behavior and discuss research extensions.

## Project Map

| Project | Biological framing | Input data type | Model family | Demo notebook | Executed notebook | Executed guide |
|---|---|---|---|---|---|---|
| `project_linear` | Cancer vs benign classification from expression-like features | Numeric feature matrix (`N x F`) | Logistic Regression | `project_linear/linear_demo.ipynb` | `project_linear/_executed_linear.ipynb` | `project_linear/_executed_linear.md` |
| `project_forest` | Pathogenic vs benign variant effect prediction | Tabular variant features | Random Forest | `project_forest/rf_demo.ipynb` | `project_forest/_executed_forest.ipynb` | `project_forest/_executed_forest.md` |
| `project_cnn` | Promoter-like motif detection in DNA | One-hot encoded sequence tensors | 1D CNN | `project_cnn/cnn_dna_demo.ipynb` | `project_cnn/_executed_cnn.ipynb` | `project_cnn/_executed_cnn.md` |
| `project_transformer` | Protein family classification from sequence patterns | Tokenized amino-acid sequences | Transformer Encoder | `project_transformer/transformer_protein_demo.ipynb` | `project_transformer/_executed_transformer.ipynb` | `project_transformer/_executed_transformer.md` |

## Why This Structure Works for Teaching

- Same end-to-end structure across four model families.
- Different biological modalities: tabular, DNA sequence, protein sequence.
- Fast runtime, so participants can run and modify code live.
- Clear transition from interpretable baselines to deep sequence models.

## Run Everything Locally

From `C:\Users\tanwa\Workshop`:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Open each notebook and run all cells in order.

Optional: execute notebooks headlessly for reproducibility:

```powershell
.\.venv\Scripts\python.exe -m pip install nbconvert nbformat ipykernel
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=180 --output _executed_linear.ipynb project_linear/linear_demo.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=180 --output _executed_forest.ipynb project_forest/rf_demo.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=180 --output _executed_cnn.ipynb project_cnn/cnn_dna_demo.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=180 --output _executed_transformer.ipynb project_transformer/transformer_protein_demo.ipynb
```

## Teaching Flow Suggestion (90-120 minutes)

1. Start with `project_linear`: model interpretation and baseline thinking.
2. Move to `project_forest`: nonlinear interactions in variant tables.
3. Move to `project_cnn`: raw DNA to tensor and motif learning.
4. Finish with `project_transformer`: tokenization, positional encoding, self-attention.

## Important Workshop Message

The main objective is not synthetic benchmark performance. The objective is to teach participants the engineering pattern that lets them convert their own biological data into deployable, testable AI workflows.
