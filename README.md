# Hydrogen Failure Prediction

### 🔧 Model Tuning & Evaluation

To improve the model's ability to detect failures, I used `GridSearchCV` to optimize hyperparameters in a `RandomForestClassifier`, with a focus on maximizing **recall** for the minority class (`failure_event = 1`), which is critical in safety-related systems like hydrogen fuel cells.

**Tuning Details:**
- Parameters tested:
  - `n_estimators`: [100, 200]
  - `max_depth`: [None, 10, 20]
  - `min_samples_split`: [2, 5]
  - `class_weight`: [None, 'balanced', {0:1, 1:3}]
- Scoring metric: **Recall**
- Cross-validation folds: 5

**Results:**
- Best estimator: `RandomForestClassifier(min_samples_split=5, class_weight='balanced')`
- However, this tuned model did **not significantly outperform** the original in recall or F1-score for the failure class.
- Given this, I kept the **original model** for final deployment, but the tuning attempt is documented here for transparency.

