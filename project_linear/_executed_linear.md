# Executed Notebook Guide: Linear Baseline

Source notebook: `linear_demo.ipynb`  
Executed artifact: `_executed_linear.ipynb`

This guide explains exactly what happens in the executed notebook, with emphasis on input data contracts, method choices, and function-level behavior.

## 1) Problem Framing

Task: classify tumor samples into `malignant (0)` vs `benign (1)` from numeric expression-like features.

Why this notebook exists:

- It is a baseline model with high interpretability.
- It establishes a reproducible benchmark before nonlinear models.
- It teaches standard habits that transfer to RNA-seq and other omics tables.

## 2) Input Data Details

Data source in notebook:

```python
breast = load_breast_cancer()
X = breast.data
y = breast.target
```

Input schema:

- `X`: 2D numeric matrix, shape `(569, 30)`.
- `y`: 1D binary labels, shape `(569,)`.
- `feature_names`: list of 30 named features used later for interpretation.

Train/test split contract:

- `test_size=0.2` gives 455 train and 114 test examples.
- `stratify=y` preserves class ratios in both splits.

## 3) Method Summary

Pipeline used in the notebook:

1. Load data.
2. Split train/test.
3. Standardize features with `StandardScaler`.
4. Fit logistic regression.
5. Evaluate with accuracy and confusion matrix.
6. Interpret coefficients and visualize strongest features.

Why this method is suitable for teaching:

- Fast and deterministic.
- Easy to inspect mathematically.
- Coefficients can be mapped back to biological hypotheses.

## 4) Cell-by-Cell Walkthrough

### Code cell 1: Imports and reproducibility

- Imports NumPy, pandas, matplotlib, and core sklearn components.
- Defines `RANDOM_STATE = 42`.
- Calls `np.random.seed(RANDOM_STATE)`.

Key point: deterministic runs reduce confusion during live workshops.

### Code cell 2: Data loading

- Calls `load_breast_cancer()`.
- Extracts `X`, `y`, and `feature_names`.
- Prints shapes and class names.

Executed output confirms:

- `X shape: (569, 30)`
- `y shape: (569,)`

### Code cell 3: Split and scaling

- Calls `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`.
- Instantiates `StandardScaler()`.
- Fits scaler only on train data.
- Transforms both train and test data.

Executed output confirms:

- Train shape `(455, 30)`
- Test shape `(114, 30)`

Data leakage note:

- Correct pattern is used: `fit` on train only, then `transform` on test.

### Code cell 4: Model fitting and basic metric

- Creates `LogisticRegression(max_iter=1000, random_state=42)`.
- Fits on scaled train data.
- Predicts on scaled test data.
- Computes `accuracy_score`.

Executed output:

- `Test accuracy: 0.982`

### Code cell 5: Confusion matrix

- Computes `confusion_matrix(y_test, y_pred, labels=[0, 1])`.
- Uses `ConfusionMatrixDisplay(...).plot(...)`.

Why important:

- Biomedical models must inspect class-specific errors, not only aggregate accuracy.

### Code cell 6: Coefficient interpretation

- Reads linear weights from `clf.coef_[0]`.
- Creates dataframe with feature names and coefficient values.
- Adds absolute value column and sorts descending.
- Selects top 10 features by absolute coefficient magnitude.

Interpretation rule:

- Sign indicates direction of association with the positive class.
- Magnitude indicates influence in standardized feature space.

### Code cell 7: Coefficient plot

- Plots top coefficients with horizontal bars.
- Uses color by sign (`red` positive, `blue` negative).
- Inverts y-axis for ranking display.

## 5) Function/API Explanation

No custom `def` or `class` is defined in this notebook. The key learning is how standard APIs compose into a full pipeline.

| API | What it does | Why used here |
|---|---|---|
| `load_breast_cancer()` | Returns feature matrix and labels | Provides a clean stand-in for expression-like tabular data |
| `train_test_split(...)` | Creates holdout split | Required for honest out-of-sample evaluation |
| `StandardScaler` | Centers/scales each feature | Critical for stable and interpretable logistic coefficients |
| `LogisticRegression` | Fits linear decision boundary | Strong interpretable baseline model |
| `accuracy_score` | Computes correct prediction rate | Quick first performance check |
| `confusion_matrix` | Counts TP/TN/FP/FN by class | Reveals error types hidden by accuracy |
| `clf.coef_` | Exposes learned linear weights | Core interpretability output |

## 6) Executed Outputs to Discuss with Participants

- High held-out accuracy (`0.982`) demonstrates baseline strength.
- Confusion matrix shows specific error locations.
- Top coefficient table and bar chart support feature-level reasoning.

## 7) How to Adapt for Real Research Data

Minimal code substitutions:

1. Replace `load_breast_cancer()` with loading your own matrix and labels from `data/`.
2. Keep the same split, scaling, fit, predict, evaluate sequence.
3. Pass real gene/protein feature names into the coefficient table.

Recommended additions for research-grade use:

- Cross-validation.
- ROC-AUC and PR-AUC.
- Confidence intervals via repeated splits.
- Batch-effect aware preprocessing.

## 8) Key Teaching Message

This notebook demonstrates a full bioinformatics ML loop with explicit interpretability. Participants should treat this as the reference baseline they compare every advanced model against.
