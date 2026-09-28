# Model Registry

A registry that cannot say no is not a control. The registry exists to make
"the backtest looked good, so it goes live" an unreachable state.

---

## The promotion ladder

```
EXPERIMENT ─► TRAINED ─► VALIDATED ─► APPROVED ─► PAPER_TRADING ─► LIVE_ELIGIBLE
                                                                            │
                                                                          ▼
                                                                        RETIRED
```

Training produces a model at `TRAINED` and nothing more. Nothing moves forward
on its own.

```python
registry = ModelRegistry()
registry.register(model)                            # -> TRAINED
```

### Each transition is enforced

```python
registry.promote("rf1", ModelStage.APPROVED)
# ModelPromotionError: cannot promote rf1@1.0.0 from trained to approved:
#   a model advances one stage at a time through experiment -> trained ->
#   validated -> approved -> paper_trading -> live_eligible. Skipping
#   validation or approval is precisely the failure mode this ladder exists
#   to prevent.

registry.promote("rf1", ModelStage.VALIDATED)
# ModelPromotionError: promotion to VALIDATED requires a ValidationRecord.
#   Training is not validation; a model must be evaluated on data it never
#   saw before it can be considered validated.
```

| Transition | Precondition |
|---|---|
| → `VALIDATED` | A `ValidationRecord` whose checks all pass |
| → `APPROVED` | At least one passed validation; records who approved and when |
| → `PAPER_TRADING` | A recorded approval, by a named person |
| → `LIVE_ELIGIBLE` | Passed validation, recorded approval, paper-trading record with authorisation, **and a locked version** |

---

## Validation records carry the evidence

```python
record = ValidationRecord(
    performed_at=utcnow(),
    performed_by="alice",
    oos_metrics={"sharpe": 1.2, "profit_factor": 1.4,
                 "max_drawdown": 0.15, "n_trades": 150},
    in_sample_metrics={"sharpe": 1.5},
    dataset_version="a3f9c2...",
    period_start=..., period_end=...,
    n_oos_observations=150,
    passed=True,
    checks={...},
)
registry.promote("rf1", ModelStage.VALIDATED, validation=record)
```

Empty metrics cannot be promoted. A record without an OOS period, a dataset
version, and an observer's name is not evidence.

### Criteria

`ValidationCriteria` encodes the minimum:

| Criterion | Default | Why it exists |
|---|---|---|
| `min_oos_observations` | 100 | A Sharpe of 3 over 40 observations is noise |
| `min_oos_sharpe` | 0.5 | A floor, not a target |
| `min_oos_profit_factor` | 1.05 | Must exceed 1.0 plus a margin |
| `max_oos_drawdown` | 0.35 | Tolerable pain |
| `max_is_oos_sharpe_degradation` | 0.6 | The overfitting signature |
| `min_trades` | 20 | Below this, trade statistics are undefined |
| `require_monotonic_calibration` | `True` | An unorderable score cannot size positions |

Every check records **whether it passed, the observed value, and the
threshold** — so a rejected promotion shows exactly which bar was missed rather
than a bare `False`.

```python
passed, checks = evaluate_validation(oos, in_sample, calibration_is_monotonic=True)
# checks["min_oos_sharpe"] == {"passed": False, "observed": 0.2, "required": 0.5}
```

### A missing calibration measurement fails the check

```python
evaluate_validation(oos, in_sample, calibration_is_monotonic=None)
# checks["calibration_monotonic"] ==
#   {"passed": False, "observed": "not measured",
#    "required": "measured out of sample"}
```

"Unknown" is not "passed".

### No losing trades is not a pass

An undefined profit factor — no losing trades at all — is treated as a **failed**
check, not an infinite one. `NaN` is not evidence of quality; it is an absence of
evidence.

---

## Version immutability

```python
registry.register(model)                                  # ok
registry.register(refit_of_same_id_and_version)           # identical: returns existing
registry.register(different_model_same_id_and_version)    # -> ModelVersionError
# rf1@1.0.0 is already registered with a different fingerprint. A registered
# version is immutable; create a new version instead so that earlier results
# remain reproducible.
```

A silently-changed model invalidates every backtest that referenced it. The
fingerprint covers the model id, version, parameters, seed, feature list, and
manifest hash.

```python
registry.versions("rf1")        # ['1.0.0', '1.1.0']
registry.get("rf1", "1.0.0")    # the exact object that was validated
```

---

## Locking

```python
registry.lock("rf1")
```

Freezes the *definition*, not the lifecycle position. A locked model can still
be promoted to `LIVE_ELIGIBLE` — locking is a **prerequisite** for that, so
refusing the final promotion because the model is locked would make live
eligibility unreachable. Retiring is likewise always permitted.

Locking is only permitted from `APPROVED` onward; locking a model that has not
been approved would let an unvalidated model be frozen in place as though it
were blessed.

---

## The registry cannot authorise trading

```python
registry.promote("rf1", ModelStage.LIVE_ELIGIBLE, authorisation_id="AUTH-001")
# entry.live_authorisation_id == "AUTH-001"
```

What this does: marks a model eligible, and records which authorisation the
operator claims covers it.

What this does **not** do: grant permission to trade. The authorisation itself
is a signed artefact verified by
[`quantforge.execution.live_gate`](LIVE_TRADING.md), and role/MFA checks are
enforced separately. There is no code path through this module that authorises
an order.

That separation is deliberate: the thing that decides whether a model may trade
real money must not be the thing that trained it.

---

## Drift

```python
registry.record_drift("rf1", DriftType.FEATURE_DRIFT, {"psi": 0.41})
# action_taken: "alerted; no automatic retrain or deployment"
```

Detection **raises an alert for a human** and changes nothing else. The model's
stage is untouched.

Auto-retraining on drift tends to lock in whatever regime caused the drift,
which is the opposite of what is wanted. A model that silently retrains on
changed data is a model nobody validated.

### Detection methods

| Method | Measures | Reference |
|---|---|---|
| `detect_feature_drift` | Per-column PSI | The **training** distribution |
| `detect_prediction_drift` | PSI of the model's own output distribution | Training-period scores |
| `detect_calibration_drift` | Brier-score deterioration between two OOS windows | The earlier OOS window |

PSI conventions: `< 0.1` no meaningful shift, `0.1–0.25` moderate, `> 0.25`
significant.

Comparing two *out-of-sample* windows measures regime change, not model
degradation. The reference must be the training distribution, or the number is
describing something other than what it claims.

`detect_calibration_drift` reports a rising Brier score as a statement about
probability accuracy, explicitly not as a claim that the model is losing money.
The two are different things and conflating them produces bad decisions.

---

## Retiring

```python
registry.retire("rf1", reason="replaced by 1.2.0 after drift review")
```

`stage` becomes `RETIRED`, `retired_at` is stamped, and the reason is appended
to the notes. The model object remains queryable — retiring is not deleting, and
the record of what was live and when is part of the audit trail.

---

## Audit

Every registry mutation fires `audit_hook`:

```python
registry.audit_hook = audit.record   # or a function
```

Events: `model_registered`, `model_promoted`, `model_locked`, `model_retired`,
`drift_detected`, `paper_trading_recorded`. Combined with the hash-chained
audit log, this gives a complete, tamper-evident history of every model's path
from an experiment to production or retirement.

---

## What the registry does not do

- It does not evaluate models. That is [MACHINE_LEARNING.md](MACHINE_LEARNING.md).
- It does not run backtests or paper trading. It records that they happened.
- It does not monitor anything. Drift detection raises alerts; monitoring and
  alerting are [EXECUTION.md](EXECUTION.md).
- It does not enforce regulatory approval or suitability. Out of scope.
- It does not prevent a determined operator from approving a weak model. It
  prevents skipping the steps, not from making a bad judgement within a step.
