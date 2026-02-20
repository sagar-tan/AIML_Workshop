# Executed Notebook Guide: Random Forest Variant Classifier

Source notebook: `rf_demo.ipynb`  
Executed artifact: `_executed_forest.ipynb`

This guide documents the executed notebook in a teaching-friendly format, including synthetic input design, nonlinear method rationale, and every major function call.

## 1) Problem Framing

Task: classify sequence variants as `pathogenic (1)` or `benign (0)` from tabular features.

Real-world analogy:

- Each row corresponds to one candidate variant.
- Each column is a computed annotation feature.
- Model predicts clinical risk class.

## 2) Input Data Details

The notebook synthesizes a structured variant table with 1200 rows and 4 features.

Feature columns:

1. `conservation_score`: uniform in `[0, 1]`
2. `aa_change_index`: normal distribution
3. `hydrophobicity_change`: normal distribution
4. `structural_impact`: uniform in `[0, 1]`

Label generation uses a nonlinear logit:

```python
logit = (
    4.0 * (conservation_score - 0.6)
    + 4.0 * (structural_impact - 0.6)
    + 2.0 * (conservation_score * structural_impact - 0.35)
    + 0.8 * np.abs(aa_change_index)
    - 0.3 * hydrophobicity_change
)
p = 1 / (1 + np.exp(-logit))
y = (rng.uniform(0, 1, size=n) < p).astype(int)
```

Important design point:

- The multiplicative term `conservation_score * structural_impact` encodes interaction behavior that linear models cannot represent directly.

## 3) Method Summary

Pipeline in order:

1. Generate synthetic feature table.
2. Sample labels from probabilistic nonlinear rule.
3. Split train/test with stratification.
4. Fit `RandomForestClassifier`.
5. Predict held-out labels.
6. Evaluate with accuracy and confusion matrix.
7. Inspect `feature_importances_`.

## 4) Cell-by-Cell Walkthrough

### Code cell 1: Imports and seeded RNG

- Imports NumPy, pandas, matplotlib, sklearn split/model/metrics.
- Sets `RANDOM_STATE = 42` and `rng = np.random.default_rng(RANDOM_STATE)`.

### Code cell 2: Synthetic table generation

- Creates 1200 examples of each feature.
- Computes nonlinear risk score (`logit`) and maps to probabilities with sigmoid.
- Samples binary labels probabilistically.
- Builds matrix `X` by column stacking feature arrays.
- Creates dataframe and prints preview plus class balance.

Executed outputs include:

- Printed head of feature table.
- Printed pathogenic class rate.

### Code cell 3: Train/test partition

- Uses 80/20 split with stratification.

Executed output:

- Train shape `(960, 4)`
- Test shape `(240, 4)`

### Code cell 4: Model fit and accuracy

- Initializes `RandomForestClassifier` with:
- `n_estimators=200`
- `max_depth=None`
- `n_jobs=-1`
- `random_state=42`
- Fits on train data and predicts test labels.
- Computes accuracy.

Executed output:

- `Test accuracy: 0.733`

### Code cell 5: Confusion matrix visualization

- Builds confusion matrix with explicit label ordering `[0, 1]`.
- Plots matrix for benign/pathogenic classes.

### Code cell 6: Feature importance analysis

- Extracts `rf.feature_importances_`.
- Creates sorted dataframe.
- Displays table and bar plot.

Executed ranking observed:

1. `conservation_score`
2. `structural_impact`
3. `aa_change_index`
4. `hydrophobicity_change`

## 5) Function/API Explanation

No custom functions are defined. The notebook is intentionally focused on core APIs and model behavior.

| API | Behavior | Teaching value |
|---|---|---|
| `np.random.default_rng` | Reproducible random generator | Ensures participants see consistent synthetic data |
| `np.exp` sigmoid transform | Converts logit to probability | Shows probabilistic label generation process |
| `np.column_stack` | Assembles features into matrix | Mirrors real variant table assembly |
| `train_test_split(..., stratify=y)` | Balanced split | Avoids misleading class distribution drift |
| `RandomForestClassifier` | Ensemble of decision trees | Captures nonlinear interactions automatically |
| `accuracy_score` | Scalar performance | Simple baseline metric |
| `confusion_matrix` | Error-type counts | Critical for medical risk discussion |
| `feature_importances_` | Global importance scores | Useful first-pass interpretation |

## 6) Interpretation Notes for Workshop

- Accuracy is moderate because labels are probabilistic and noisy by design.
- Feature importances recover the intended signal structure.
- This model class captures nonlinear effects without manual interaction terms.

## 7) Adaptation to Real Variant Pipelines

Practical substitutions:

1. Load real variant annotations from `data/` as a dataframe.
2. Keep numeric feature matrix extraction logic.
3. Preserve train/test split and confusion-matrix reporting.

Research-strength upgrades:

- Use cross-validation and out-of-fold evaluation.
- Add calibrated probabilities.
- Add threshold tuning for recall-sensitive workflows.
- Add SHAP for local variant-level explanations.

## 8) Key Teaching Message

This notebook demonstrates the jump from linear assumptions to interaction-aware modeling while preserving straightforward tabular workflow patterns that bioinformatics teams can adopt quickly.
