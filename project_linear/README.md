# Disease Classification from Expression-like Features (Linear Baseline)

This project teaches the most important first step in bioinformatics ML: build a strong, interpretable linear baseline before using more complex models.

## Learning Objectives

- Map a biological classification problem into `X` and `y` tensors.
- Apply a reproducible train/test split.
- Understand when and why feature standardization is required.
- Train logistic regression for binary disease classification.
- Interpret learned coefficients as directional feature effects.

## Files

- `linear_demo.ipynb`: source notebook used in teaching.
- `_executed_linear.ipynb`: executed notebook with outputs.
- `_executed_linear.md`: detailed cell-by-cell and function-level explanation for participants.
- `data/`: location for your own expression matrix and sample labels.

## Input Data (Current Demo)

The notebook uses `sklearn.datasets.load_breast_cancer()` as a clean stand-in for expression-like measurements.

- Samples: `569`
- Features per sample: `30`
- Input tensor shape: `X.shape = (569, 30)`
- Label vector shape: `y.shape = (569,)`
- Labels: `0 = malignant`, `1 = benign`

Why this is appropriate for class:

- No file I/O or preprocessing blockers.
- Numeric features already prepared.
- Lets participants focus on core ML pipeline decisions.

## Method (Notebook Execution Flow)

1. Set reproducibility controls with `RANDOM_STATE = 42` and `np.random.seed`.
2. Load feature matrix and labels with `load_breast_cancer()`.
3. Split data using `train_test_split(..., test_size=0.2, stratify=y)`.
4. Standardize features via `StandardScaler` fit on train only.
5. Train `LogisticRegression(max_iter=1000, random_state=42)`.
6. Evaluate using accuracy and confusion matrix.
7. Extract `clf.coef_[0]` to rank most influential features.
8. Plot top coefficients with sign-based color coding.

## Function and API Reference

The notebook defines no custom Python functions, so learning centers on core library calls.

| Call | Role in pipeline | Why it matters biologically |
|---|---|---|
| `load_breast_cancer()` | Loads matrix + labels | Mimics a biomarker table with known clinical labels |
| `train_test_split(..., stratify=y)` | Creates reproducible train/test partitions | Preserves class ratio, reducing misleading evaluation |
| `StandardScaler().fit_transform(X_train)` | Normalizes feature scales | Prevents magnitude-dominant features from biasing coefficients |
| `LogisticRegression(...)` | Fits linear decision boundary in feature space | Baseline for disease-vs-control classification |
| `accuracy_score(y_test, y_pred)` | Scalar performance metric | Quick sanity check before deeper diagnostics |
| `confusion_matrix(...)` | Error decomposition by class | Critical in biomedical settings with asymmetric error costs |
| `clf.coef_` | Per-feature signed weights | Supports biological interpretability and hypothesis generation |

## Last Executed Results

From `_executed_linear.ipynb`:

- Train set shape: `(455, 30)`
- Test set shape: `(114, 30)`
- Test accuracy: `0.982`
- Additional outputs:
- Confusion matrix plot.
- Top-10 absolute coefficient table.
- Horizontal bar chart of strongest coefficients.

## How to Connect This to Real Bioinformatics Data

1. Replace `load_breast_cancer()` with a matrix loaded from `data/`.
2. Keep data shape contract: rows are samples, columns are features.
3. Ensure labels are binary integers `{0,1}` for this notebook version.
4. Keep standardization and stratified splitting unchanged.
5. Add feature naming from your assay metadata for interpretable coefficient output.

## Teaching Notes for Participants

- Start by explaining why linear baselines are non-negotiable.
- Emphasize that interpretability is built in, not added later.
- Show coefficient signs as directional biological associations.
- Contrast this with black-box models that may increase accuracy but reduce explainability.

## Suggested Next Iterations

1. Add L1 regularization to induce sparse signatures.
2. Add repeated stratified splits for stability analysis of top features.
3. Add ROC-AUC and precision-recall metrics for class-imbalance scenarios.
4. Replace toy data with RNA-seq counts after log transform and batch correction.
