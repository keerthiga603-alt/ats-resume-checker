# Evaluation Report

**Model:** XGBoost
**Trained:** 2026-08-15T16:33:25.482926+00:00
**Training samples:** 17501
**Validation samples:** 3750
**Test samples:** 3751

## Validation metrics (all candidate models)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| LogisticRegression | 0.7803 | 0.8071 | 0.7365 | 0.7702 | 0.8557 |
| RandomForest | 0.7973 | 0.7794 | 0.8293 | 0.8036 | 0.8867 |
| GradientBoosting | 0.7963 | 0.7833 | 0.8192 | 0.8008 | 0.8882 |
| XGBoost | 0.8013 | 0.7952 | 0.8117 | 0.8034 | 0.8885 |

**Selected model (highest validation ROC-AUC): XGBoost**

## Held-out test set metrics (evaluated once, after model selection)

| Metric | Value |
|---|---|
| accuracy | 0.8041 |
| precision | 0.7969 |
| recall | 0.8160 |
| f1 | 0.8063 |
| roc_auc | 0.8885 |

Confusion matrix: [[1486, 390], [345, 1530]] (rows=true, cols=predicted, order=['not_compatible(0)', 'compatible(1)'])

## Feature importance

- `tfidf_cosine_similarity`: 0.4458
- `resume_length`: 0.1206
- `keyword_overlap`: 0.0856
- `skill_overlap`: 0.0832
- `technical_keyword_density`: 0.0824
- `required_skill_coverage`: 0.0774
- `education_alignment`: 0.0558
- `experience_alignment`: 0.0493
- `preferred_skill_coverage`: 0.0000
- `job_title_similarity`: 0.0000
- `relevant_section_coverage`: 0.0000