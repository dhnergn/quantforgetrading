# Live Trading

**Live trading is disabled by default and cannot be enabled by configuration.**

```python
from quantforge.execution.live_gate import LiveTradingGate
permitted, reason = LiveTradingGate().check()
# False — "no live-trading authorisation found at .../live_authorisation.json.
#          Live trading is disabled, which is the default and intended state."
```

Setting `ExecutionConfig.live_trading_enabled = True` is **not sufficient**.
`Config.validate()` rejects it unless `sandbox_only` is also deliberately
cleared, and even then the gate refuses without a signed record.

## Why a file and not a flag

A flag is a value that some future deploy, config edit, or default change can
flip. A signed record is an artefact a person produced after reviewing a
checklist, and whose signature is verifiable afterwards.

The asymmetry matters. The cost of accidentally enabling live trading is
unbounded; the cost of the ceremony is a few minutes.

The file is written **read-only (0400)**. A running process that can rewrite its
own authorisation can grant itself permission to trade. The operator creating
the record can of course replace it.

---

## The eleven confirmations

Every one must be explicitly confirmed. There is no way to produce a record
with preconditions inferred or assumed.

| Key | Requirement |
|---|---|
| `broker_connected` | Sandbox connection established, credentials validated against a **sandbox** account |
| `credentials_validated` | Keys confirmed read/trade-only, withdrawal permission removed, stored in the secret manager |
| `risk_limits_configured` | Position, sector, gross, net, leverage, drawdown, daily-loss limits reviewed by the risk manager |
| `kill_switch_configured` | Global, portfolio, strategy, and broker switches configured with tested triggers |
| `market_data_validated` | Live feed connected, staleness thresholds set, feed-versus-broker prices compared |
| `position_reconciliation_passed` | Internal positions reconciled exactly against broker positions |
| `strategy_version_locked` | The exact version validated in backtest and paper trading is locked |
| `model_version_locked` | The exact model version is locked and live-eligible |
| `paper_trading_reviewed` | Paper trading ran for a documented period, reviewed and signed off |
| `live_permission_granted` | A person with `CAN_ENABLE_LIVE_TRADING` granted permission by name |
| `explicit_confirmation` | A human typed an explicit phrase acknowledging real money is at risk |

---

## Every refusal path

| Condition | Result |
|---|---|
| No file | refused |
| Unreadable file | refused |
| `authorised: false` | refused |
| Invalid signature (wrong secret) | refused |
| Tampered record | refused |
| Expired | refused |
| Missing any confirmation | refused, listing which |
| Running config differs from the authorised one | refused, listing the mismatch |

```python
gate.check(expected_strategy_version="3.0")   # authorised for 2.0 -> refused
gate.check(expected_strategy_version="2.0")   # matches -> permitted
```

The mismatch check is what makes a specific authorisation a *specific*
authorisation. Without it, an approval for one strategy version would silently
cover a different one.

---

## Three independent barriers

1. **The signed file** — cryptographic, tamper-evident, expiring.
2. **Role and MFA** — `require_live_trading_authority` demands the permission,
   a role on the `LIVE_TRADING_ROLES` allow-list, **and** MFA. A stolen
   password alone cannot trade real money.
3. **The model registry** — cannot grant permission. It marks a model
   `live_eligible` only after validation, approval, paper trading, and a lock.
   It has no code path that authorises trading.

The API endpoint is a fourth: it requires a literal confirmation phrase.

```python
LIVE_CONFIRMATION_PHRASE = "I UNDERSTAND THIS PLACES REAL ORDERS WITH REAL MONEY"
```

---

## Operator procedure

```python
import os
from quantforge.execution.live_gate import (
    AUTHORISATION_CHECKLIST, LiveTradingGate, build_authorisation,
)
from quantforge.core.timestamps import utcnow

secret = os.environ["QUANTFORGE_LIVE_SECRET"]        # from the secret manager
authorisation = build_authorisation(
    authorised_by="alice.chen",                       # a person, by name
    environment="production",
    portfolio_id="PF_MAIN",
    strategy_version="1.4.2",                         # the locked version
    model_version="2.1.0",                           # the locked version
    risk_limits_hash=risk.limits.fingerprint(),       # the exact limits
    broker_account_id="ACC-12345",
    expires_in_days=30,
    confirmed=[k for k, _ in AUTHORISATION_CHECKLIST],
)
authorisation.sign(secret.encode())
authorisation.save()                                 # 0400, read-only

LiveTradingGate().check()                            # -> (True, "authorised by ...")
```

Rotation is a matter of replacing the file; the previous authorisation ceases
to verify the moment the secret or content changes.

---

## Before you ever get here

The gate is the last control, not the first. A working authorisation record
means somebody asserted that all of this was already done:

- the strategy was validated out of sample, not just in sample,
- the model passed the registry ladder with a real OOS period,
- paper trading ran long enough to observe the strategy's actual behaviour
  under real fills and real rejections,
- execution was tested against a **broker sandbox** with real order flow,
- the strategy's edge survives conservative cost assumptions,
- the kill switches have been triggered deliberately, in a test, and verified
  to halt.

If any of that is not true, an authorisation record is a piece of paperwork
rather than a control.

---

## What the gate does not do

- It does not verify that the human was **correct** about the confirmations.
  It verifies that a human made a claim.
- It does not monitor the strategy after activation. Drift detection and the
  kill switches do that.
- It does not prevent an authorised operator from authorising a bad strategy.
  That is what the risk controls and the review process are for.
- It does not replace regulatory compliance, suitability assessment, or
  operational due diligence. Those are outside the platform's scope.
