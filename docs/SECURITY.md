# Security

## Defence in depth

No single control here is a security boundary on its own. The design goal is
that an attacker must defeat several independent layers simultaneously, and
that a bug in one layer fails closed.

---

## Authentication

Passwords use **argon2id** when available, falling back to PBKDF2-HMAC-SHA256
at 600,000 iterations. The active algorithm is always reported rather than
silently downgraded.

```python
hasher = PasswordHasher()
hasher.algorithm          # "argon2id" or "pbkdf2_sha256"
hasher.hash("correct-horse-battery-staple")
hasher.verify(password, stored)
```

Passwords under 12 characters are rejected. A short password on a system with
broker access is a liability, not a convenience.

A malformed stored hash fails verification rather than crashing the login path,
and never authenticates.

---

## Tokens

HMAC-signed, with expiry, revocation, and a minimum 32-byte secret:

```python
TokenService(b"k" * 40, ttl_seconds=900)
token, payload = ts.issue(user)
ts.verify(token)         # constant-time signature comparison
ts.revoke(payload.token_id)
```

Short secrets are rejected at construction — a short signing key is forgeable,
and a forged token is a full authentication bypass.

Signature comparison uses `hmac.compare_digest`; a timing leak there reveals
the signature.

---

## Authorization

**Deny by default.** No permission is implied by authentication. A
successfully authenticated user with no role has no permissions.

| Role | Key permissions |
|---|---|
| `READONLY` | none |
| `RESEARCHER` | research, backtest |
| `QUANT_DEVELOPER` | + train models, paper trade, configure risk |
| `RISK_MANAGER` | configure risk, kill switch, view audit |
| `PORTFOLIO_MANAGER` | + enable live, submit/cancel live orders |
| `ADMINISTRATOR` | everything |
| `SECURITY_ADMINISTRATOR` | audit, secrets, platform admin |

Senior roles inherit junior permissions, so adding a permission to a junior
role does not silently exclude it from a senior one.

### Fails closed on error

```python
class BrokenUser:
    def has(self, p): raise RuntimeError("authorisation store unavailable")

require_permission(BrokenUser(), Permission.CAN_RESEARCH)
# AuthorizationError: permission check raised RuntimeError; denying by default
```

A bug in the authorization layer must never become an authorization bypass.

### Live trading needs more than a permission

`require_live_trading_authority` demands the permission, a role on the
`LIVE_TRADING_ROLES` allow-list, **and** MFA. Roles holding
`CAN_ENABLE_LIVE_TRADING` but not on the allow-list are refused, so granting a
role can never accidentally grant the ability to trade real money.

---

## BOLA / IDOR prevention

Every endpoint validates ownership, not just role:

```python
job = jobs[job_id]
if job["submitted_by"] != user.username and not user.has(Permission.CAN_VIEW_AUDIT):
    raise HTTPException(403, "this job belongs to another user")
```

A user cannot read another user's backtest by guessing an id. Brokers,
portfolios, and strategies are scoped the same way.

---

## Rate limiting

Token bucket per `(subject, resource)`, with a burst allowance because
interactive clients legitimately send short clusters.

| Resource | Limit |
|---|---|
| `auth` | 5/min |
| `market_data` | 60/min |
| `backtest` | 1/hour |
| `ml_training` | 2 |
| `inference` | 100/min |
| `order_submit` | 10/min |
| `order_cancel` | 5/min |
| `strategy` | 3/hour |
| `live_enable` | 2 |

Authentication failures additionally drive a lockout with a 15-minute reset
window.

Rate limits are also applied to *broker* and *order submission* operations
inside the execution layer, not only at the HTTP boundary.

---

## API hardening

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'`
- `Referrer-Policy: no-referrer`
- request body size limit
- CORS restricted to an explicit origin allow-list, never `*` with credentials
- `X-RateLimit-*` response headers
- research-heavy operations return a job id rather than blocking a worker

---

## Secret management

Broker credentials are never written to a frontend, a log, browser storage,
client-side JavaScript, a model prompt, or a strategy sandbox. They come from
the environment, or from a secret manager in production.

Prefer broker permissions that cannot withdraw funds. The authorisation
checklist requires confirming that withdrawal permission has been removed.

---

## Audit trail

Append-only, hash-chained, HMAC-signed:

```python
log.record("alice", AuditAction.ORDER_CREATED, "order/abc", {...})
log.verify_chain()
# {"valid": True, "n_entries": 42, "head_hash": "e1361b1...", ...}
```

Each entry includes the hash of the previous one, so modifying an early entry
invalidates every later hash:

```python
log._entries[1].detail = {"sharpe": 99.0}
log.verify_chain()["broken_at_sequence"]   # 2
```

Audited: strategy, model, feature, dataset, backtest, broker connection,
credential change, risk-limit change, paper and live activation, order
creation/cancellation/fill/rejection, position change, and kill-switch events.

The chain is a tamper *detector*, not a signature of authorship. For
non-repudiation the HMAC provides the second half.

---

## Strategy sandbox

### Explicitly not a security boundary

In-process limits can be defeated by sufficiently determined code. Running
untrusted strategies requires container or microVM isolation with seccomp, a
read-only root filesystem, no network namespace, and cgroup limits enforced
outside the interpreter. The two together are what make third-party code
defensible.

`StrategySandbox.describe()["boundary_note"]` states this in the returned
payload, so it cannot be quietly relied upon as a sole control.

### What it does provide

AST-based screening, not regex — a regex cannot distinguish a docstring from
an import, nor an attribute named `system` from a call to it.

```python
sandbox.check_source("import socket")        # blocked
sandbox.check_source("from subprocess import run")  # blocked
sandbox.check_source("os.system('ls')")      # blocked
sandbox.check_source("eval('1+1')")          # blocked
sandbox.check_source("import numpy as np")   # allowed
```

Blocked modules cover network, database, process spawning, filesystem,
serialisation, and dynamic import. A strategy receives a feature row, never a
connection.

Resource limits: CPU seconds, wall seconds, memory, output size, open file
descriptors, and file size. The wall-clock timeout abandons the thread rather
than killing it, which is a further reason this is a guard and not a boundary.

---

## Injection defences

- **SQL:** SQLAlchemy parameter binding exclusively; no string-built queries.
- **XSS:** JSON responses, `Content-Security-Policy`, no HTML rendering of
  untrusted content.
- **SSRF:** no user-controlled URL fetching; adapters are code, not endpoints.
- **CSV/JSON parsing:** pandas and `json` module parsers, not `eval`.

---

## Supply chain

Dependency versions are recorded in every experiment manifest
(`capture_environment()`), so a result that will not reproduce can be
attributed to a library change rather than a strategy change.

---

## What is not covered

- TLS termination and certificate management (deployment concern)
- Network segmentation (deployment concern)
- Database encryption at rest (deployment concern)
- Multi-factor implementation (integrated at the IdP in production)
- Kernel hardening (deployment concern)
- Regulatory compliance (outside scope entirely)
