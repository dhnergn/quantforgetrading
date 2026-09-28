# Data Leakage

Distinct from look-ahead bias. Look-ahead is about time; leakage is about
information reaching a model that it should not have had, whether or not it
involves a future date.

## The four leaks

| Leak | Symptom | Defence |
|---|---|---|
| **Target leakage** | A feature that is the target, or derived from it | Targets built in a separate function from features |
| **Preprocessing leakage** | Scaler/PCA/imputer fitted on all data | Stateful transformers, `assert_no_future_fit` |
| **Selection leakage** | Features or hyperparameters chosen using the test set | Test set sealed; search confined to train/validation |
| **Contamination** | Test data influencing training | Strict temporal splits with an embargo |

---

## Stateful preprocessing

Every transformer separates `fit` from `transform`:

```python
scaler = TrainScaler()
scaler.fit(X_train)              # parameters come from here
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)    # train parameters, reused
X_test_scaled = scaler.transform(X_test)  # train parameters, reused
```

Using `transform` before `fit` raises:

```python
TrainScaler().transform(X)
# LeakageError: TrainScaler was used before fit(). A transformer must be
# fitted on the training period before it transforms anything.
```

### Why `df=0` in the scaler

`TrainScaler` uses population standard deviation, not sample. With `ddof=1` the
scaler itself introduces a small per-column bias on short samples. A constant
column yields scale `1.0` rather than being zero-filled, because zero-filling
destroys the information that the column exists but does not vary.

---

## Pipeline audit

`PreprocessingPipeline` records the order of every fit and transform, so a
reviewer can confirm the sequence rather than trust it:

```python
p = (PreprocessingPipeline()
     .add(DropIncompleteRows(max_drop_fraction=0.3))
     .add(TrainVarianceFilter())
     .add(TrainCorrelationFilter(threshold=0.95))
     .add(TrainScaler()))

p.fit(X_train)
X_val = p.transform(X_val)

p.audit()["step_order"]
# ['DropIncompleteRows', 'TrainVarianceFilter', 'TrainCorrelationFilter', 'TrainScaler']
```

Refitting raises. A silently-refitted pipeline re-derives parameters from
different data, which is the exact failure being prevented.

### The fit-window audit

```python
assert_no_future_fit(pipeline, train_index)
# LeakageGuardViolation: step TrainScaler was fitted on 2020-01-01 to
# 2022-04-19, which extends outside the training window.
```

---

## PCA

Fitted on the training window only. The demonstration that this matters:

```python
pca_full  = TrainPCA(n_components=2).fit(all_data)      # WRONG
pca_train = TrainPCA(n_components=2).fit(train_only)    # RIGHT

pca_full.components_  != pca_train.components_          # basis differs
```

The in-sample fit quality is *identical* in both cases, which is why leakage
cannot be detected by looking at model performance. The leak only appears
out of sample, which is the definition of an unusable model.

`TrainPCA` also rejects:

- `NaN` input, since SVD would silently propagate it into the components,
- more features than observations, where the covariance is rank-deficient and
  the components are not identifiable.

---

## Imputation

`TrainMedianImputer` uses the **training-period** median per column.

Median rather than mean because financial features are heavy-tailed: a single
crisis bar drags a mean far above any typical value, imputing a missing
observation to something that never happened.

```python
imp = TrainMedianImputer().fit(X_train)
imp.statistics["spread"]        # the training median, not the full-sample one
imp.missing_report(X_val)       # how many values this would fill
```

`DropIncompleteRows` refuses to discard more than a configured fraction:

```python
# LeakageError: dropping incomplete rows would remove 87.3% of the data.
# Choose an explicit imputation strategy instead; the resulting subsample may
# not be representative.
```

Silently proceeding would train a model on a small, non-random subsample.

---

## Feature selection

`TrainVarianceFilter` and `TrainCorrelationFilter` are fitted on training data
only. A column flat in training but varying later is dropped, which is correct:
nothing in the training sample would have taught the model to use it.

The correlation filter drops one column from each highly correlated pair.
Redundancy inflates the apparent importance of a duplicated feature and
destabilises tree splits.

> Selecting features by their relationship to the *target* is a separate and
> more dangerous step. It must also happen inside the training window, and it
> is a form of model fitting, not preprocessing.

---

## Target construction

Targets are built in a separate function with an explicit horizon, never inside
the feature frame:

```python
target = build_target(prices, horizon=5, label_thresholds=(-0.02, 0.02))
# forward_return, label, and recorded metadata:
#   thresholds: (-0.02, 0.02)
#   "Thresholds are fixed before evaluation; optimising them on the test set
#    invalidates the test."
```

A forward return placed in the feature matrix is the most direct target leak:
it produces a near-perfect in-sample model and a useless out-of-sample one.

---

## Train/validation/test separation

```python
split = temporal_split(index, train_frac=0.60, validation_frac=0.20,
                       embargo_bars=5)
split.train       # 2018-01-01 -> 2021-06-04
split.validation  # 2021-06-21 -> 2022-07-29
split.test        # 2022-08-15 -> 2023-09-29   <- touched once, at the end
```

The **embargo** matters more than it appears. A label with a 5-day horizon at
the end of training uses prices up to 5 days into validation, so the windows
share observations and validation performance is partly in-sample. The embargo
must be at least the label horizon.

`temporal_split` refuses a configuration with no test window:

```python
temporal_split(index, 0.6, 0.4)
# ValidationError: train_frac + validation_frac = 1.0 leaves no test period;
# an out-of-sample test is not optional
```

---

## Tests

`tests/test_leakage.py::TestPreprocessingLeakage`:

- transform before fit raises
- the scaler reuses training statistics
- a second `fit` raises
- a full-dataset fit is detected by the audit
- full-data and train-only PCA produce different bases
- PCA rejects `NaN` and too-few-observations
- the variance filter is fitted on train
- the imputer uses train medians
- dropping too many rows raises
