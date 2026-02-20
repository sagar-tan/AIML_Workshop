# Variant Effect Prediction with Random Forest (Tabular)

This project teaches nonlinear modeling for variant interpretation when input data is a structured feature table.

## Learning Objectives

- Build a variant-level feature matrix suitable for ML.
- Understand nonlinear feature interactions in pathogenicity prediction.
- Train and evaluate an ensemble tree model.
- Interpret feature importance output and its limits.

## Files

- `rf_demo.ipynb`: source notebook.
- `_executed_forest.ipynb`: executed notebook with outputs.
- `_executed_forest.md`: deep explanation of data generation, model fit, and each API call.
- `data/`: placeholder for real variant feature tables.

## Input Data (Current Demo)

The demo creates synthetic per-variant features to avoid external downloads while preserving realistic structure.

- Number of variants: `1200`
- Number of features: `4`
- Input tensor shape: `X.shape = (1200, 4)`
- Label shape: `y.shape = (1200,)`
- Label meaning: `1 = pathogenic`, `0 = benign`

Feature schema:

| Feature | Type | Range or distribution | Biological interpretation |
|---|---|---|---|
| `conservation_score` | continuous | uniform `[0, 1]` | Higher conservation often indicates functional constraint |
| `aa_change_index` | continuous | normal `(0, 1)` | Proxy for amino-acid substitution severity |
| `hydrophobicity_change` | continuous | normal `(0, 1)` | Proxy for biophysical shift after mutation |
| `structural_impact` | continuous | uniform `[0, 1]` | Proxy for structural perturbation risk |

Label mechanism includes nonlinearity:

- Higher risk when `conservation_score` and `structural_impact` are both high.
- Additional contribution from `abs(aa_change_index)`.
- Mild negative effect from `hydrophobicity_change`.

## Method (Notebook Execution Flow)

1. Fix randomness with `RANDOM_STATE = 42` and `np.random.default_rng`.
2. Synthesize feature vectors for 1200 mock variants.
3. Compute a nonlinear logit and convert to probability via sigmoid.
4. Sample binary labels from that probability.
5. Split with stratified 80/20 train/test partition.
6. Train `RandomForestClassifier(n_estimators=200, n_jobs=-1)`.
7. Predict on held-out test data.
8. Evaluate with accuracy and confusion matrix.
9. Rank `feature_importances_` and visualize.

## Function and API Reference

No custom functions are defined in this notebook; the teaching emphasis is on core `scikit-learn` APIs.

| Call | Role in pipeline | Practical note |
|---|---|---|
| `np.random.default_rng(42)` | Controlled random generator | Keeps synthetic data reproducible across runs |
| `np.column_stack([...])` | Builds feature matrix | Mirrors real-world engineered variant tables |
| `train_test_split(..., stratify=y)` | Robust split | Prevents class-ratio drift between train/test |
| `RandomForestClassifier(...)` | Nonlinear ensemble learner | Captures feature interactions with minimal scaling assumptions |
| `rf.predict(X_test)` | Hard-label inference | Used for confusion matrix and accuracy |
| `accuracy_score(...)` | Overall correctness | Good first metric but insufficient alone in clinical contexts |
| `confusion_matrix(...)` | Class-specific errors | Required for discussing false negatives vs false positives |
| `rf.feature_importances_` | Global feature ranking | Useful but not causal and can be biased by correlated features |

## Last Executed Results

From `_executed_forest.ipynb`:

- Train shape: `(960, 4)`
- Test shape: `(240, 4)`
- Test accuracy: `0.733`
- Importance ranking observed:
- `conservation_score` highest.
- `structural_impact` second.
- `aa_change_index` and `hydrophobicity_change` lower.

## Real-Data Adaptation Path

1. Replace synthetic generation with real annotated variant table ingestion from `data/`.
2. Keep one row per variant and explicit feature dictionary documentation.
3. Add missing-value handling and leakage checks.
4. Add cross-validation and probability calibration.
5. Add precision, recall, F1, ROC-AUC, and PR-AUC.

## Teaching Notes for Participants

- Explain why trees are a natural next step after linear baselines.
- Highlight that biological effects are often interaction-driven, not purely additive.
- Emphasize that feature importance is directional for exploration, not proof of mechanism.
- Link outputs to triage workflows where high-risk variants are prioritized for follow-up.

## Suggested Next Iterations

1. Compare random forest against gradient boosting and XGBoost.
2. Add SHAP-based interpretation for local explanations.
3. Add uncertainty thresholds to flag variants for manual review.
4. Integrate population allele frequency and domain annotations.
