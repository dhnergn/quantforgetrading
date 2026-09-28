# Machine Learning

## What the models are for

Predictive tasks that feed a deterministic strategy: return classification,
return regression, volatility estimation, asset grouping, and anomaly
detection. The models produce scores; the strategy decides; the risk engine
vetoes.

**No model here predicts markets.** A classifier's 0.7 is a model score, not a
70% likelihood of a price move. Every evaluation returns a `note` field saying
so.

---

## Preprocessing is the safety boundary

Every transformer is stateful, fitted on the training period only:

```python
pipeline = (PreprocessingPipeline()
            .add(DropIncompleteRows(max_drop_fraction=0.3))
            .add(TrainVarianceFilter())
            .add(TrainCorrelationFilter(threshold=0.95))
            .add(TrainScaler()))

pipeline.fit(X_train)
X_val = pipeline.transform(X_val)     # train parameters, reused
assert_no_future_fit(pipeline, train_index)
```

Transforming before fitting, refitting, and fitting on the full dataset all
raise. See [DATA_LEAKAGE.md](DATA_LEAKAGE.md).

---

## Model families

### Classification

```python
LogisticRegressionModel(model_id, C=1.0)
RandomForestClassifierModel(model_id, n_estimators=200, class_weight="balanced")
GradientBoostingClassifierModel(model_id, learning_rate=0.05, max_depth=3)
```

The regularised linear model is a first-class citizen, not a fallback. A tree
ensemble that cannot beat it on the same features is not earning its complexity
or its opacity.

### Regression

```python
LinearRegressionModel(model_id, alpha=1.0)     # ridge
RandomForestRegressorModel(model_id)
GradientBoostingRegressorModel(model_id)
NeuralNetworkModel(model_id, hidden_layer_sizes=(64, 32), early_stopping=True)
```

The neural network is **optional and not a default**. A high-capacity model
with many local optima is frequently beaten by a regularised linear model or a
shallow ensemble on tabular financial data. Its training curve is exposed so the
early-stopping point is visible rather than implicit.

### Clustering

```python
kmeans_cluster(X, n_clusters=5)          # reports silhouette
hierarchical_cluster(X, n_clusters=5, linkage="ward")
dbscan_anomaly(X, eps=0.5, min_samples=5)
```

Cluster assignments are time-aware when used historically: a model fitted on a
window is not silently applied to a later date. DBSCAN's `eps` is
scale-dependent, so it is only meaningful after standardisation — a high noise
fraction usually means mis-set `eps`, not anomalous data.

---

## Reproducibility

Every fit records a manifest:

```python
model.fit(X_train, y_train)
model.trained.manifest.to_dict()
# experiment_type, dataset_version, feature_set_version,
# strategy_version, random_seed, preprocessing (full pipeline audit),
# time_period, parameters
model.fingerprint()    # stable content hash
```

`set_seed` is called at every entry point. Same data and same config reproduces
the same model.

---

## Evaluation

```python
ev = model.evaluate(X_test, y_test)
ev["accuracy"], ev["macro_f1"], ev["brier_score"], ev["roc_auc"]
ev["confusion_matrix"], ev["recall_per_class"]
```

Macro F1 is included because accuracy alone is misleading on imbalanced
labels — the normal case for a directional target, where always predicting
"flat" can score well while being useless.

For regression:

```python
ev["r_squared"], ev["information_coefficient"], ev["directional_accuracy"]
ev["ic_std_error"]
```

The information coefficient (rank correlation between prediction and outcome)
matters more than R² for a directional model: R² rewards magnitude
predictions on a noisy target and is usually small.

---

## Calibration

```python
lift = model.out_of_sample_lift(X_test, y_test, buckets=5)
lift[["bucket", "n", "mean_predicted", "observed_rate"]]
```

```
 bucket  n  mean_predicted  observed_rate   lift
      0 67          0.2239         0.1642 -0.0597
      1 66          0.4507         0.4848  0.0342
      2 67          0.7531         0.5821 -0.1710
```

The most honest single check on a probability model: if the top bucket of
predicted probabilities does not have a higher observed outcome rate than the
bottom bucket, the scores cannot be used to size positions.

A well-ordered model shows `observed_rate` rising monotonically with
`mean_predicted`. Non-monotonicity is a real defect, not a curiosity.

Calibration is measured **out of sample**. It is a statement about the model's
own outputs, not a guarantee about future outcomes.

---

## Feature importance

```python
model.feature_importance(top_n=20)
```

Tree impurity importance, mean |coefficient| for linear models, or
`NaN` when the estimator exposes neither.

Every result carries:

```
interpretation: "predictive association within the training sample; not a
                  causal effect and not a guarantee of future relationship"
```

MDI importance is biased toward high-cardinality features. Permutation
importance is more reliable but slower. Neither measures causation, and
importance is not stability: a feature can be important in-sample and useless
out of sample.

---

## Model registry

```
EXPERIMENT ─► TRAINED ─► VALIDATED ─► APPROVED ─► PAPER_TRADING ─► LIVE_ELIGIBLE
```

Each promotion is enforced with a precondition:

```python
registry.promote("rf1", ModelStage.APPROVED)          # -> blocked, skipped a stage
registry.promote("rf1", ModelStage.VALIDATED)        # -> blocked, no evidence
registry.promote("rf1", ModelStage.VALIDATED, validation=record)
registry.promote("rf1", ModelStage.APPROVED, approved_by="risk_manager")
registry.promote("rf1", ModelStage.PAPER_TRADING)
registry.promote("rf1", ModelStage.LIVE_ELIGIBLE)     # -> blocked, no paper record
```

"Training is not validation" is enforced, not advised. A model must be
evaluated on data it never saw before it can be considered validated.

Registered versions are **immutable**. Registering the same
`(model_id, version)` with a different fingerprint raises, because a
silently-changed model invalidates every backtest that referenced it.

The registry can mark a model `live_eligible` but **cannot authorise trading**.
That is [LIVE_TRADING.md](LIVE_TRADING.md).

---

## Drift

```python
detect_feature_drift(train_ref, current)        # PSI per column
detect_prediction_drift(train_scores, live_scores)
detect_calibration_drift(ref, current, "p", "y") # Brier deterioration
```

Detection **raises an alert for a human**. It never triggers a retrain or a
deployment:

```python
registry.record_drift("rf1", DriftType.FEATURE_DRIFT, {"psi": 0.41})
# action_taken: "alerted; no automatic retrain or deployment"
```

Auto-retraining on drift tends to lock in whatever regime caused the drift,
which is the opposite of what is wanted.

PSI conventions: `< 0.1` no meaningful shift, `0.1–0.25` moderate, `> 0.25`
significant. The reference must be the *training* distribution; comparing two
out-of-sample windows measures regime change, not model degradation.

---

## Minimum data

```python
model.fit(X, y)   # < 30 rows
# InsufficientDataError: 12 training rows is too few for a fitted model.
#   30 is the floor; below that any result is dominated by sampling noise.
```

---

## Known limitations

- No hyperparameter search framework is included; grid and random search are
  provided as callables in [VALIDATION.md](VALIDATION.md) rather than tied to
  a specific library.
- No feature-selection-by-importance step, which is deliberate: it is a form of
  model fitting and must live inside the training window.
- Calibration uses the library's built-ins; there is no cross-fitted
  calibration, so a calibration figure measured on training data would be
  optimistic. The platform measures it out of sample.
- Class imbalance is handled by `class_weight`; no resampling strategies are
  included.
- Neural networks are CPU-only by default; GPU requires explicit configuration.
