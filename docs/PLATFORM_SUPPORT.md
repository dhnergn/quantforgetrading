# Platform Support

## Summary

| Platform | As a **host** (runs the platform) | As a **client** (talks to it) |
|---|---|---|
| **Linux** | Supported, tested | — |
| **macOS** | Supported, tested | — |
| **Windows** | Supported, tested, with one documented degradation | — |
| **iOS** | **Not supported** | Supported |
| **Android** | **Not supported** | Supported |

```bash
python scripts/bootstrap.py            # install and verify
python scripts/bootstrap.py --host-info  # what this machine supports
```

---

## Why iOS and Android are not hosts

This is a platform constraint, not a missing feature, and no amount of porting
removes it.

**There are no binary wheels for the compute stack on iOS.** numpy, pandas,
scipy, and scikit-learn all ship compiled extensions built against specific
platform ABIs. There is no `ios_*` platform tag in the wheel ecosystem for
them. On Android, numpy and pandas can be built via Chaquopy or Briefcase, but
**scikit-learn has no Android wheel**, and the platform depends on it.

**iOS additionally forbids the mechanism those libraries need.** Since iOS 14,
third-party code may not be JIT-compiled. numpy and scikit-learn rely on JIT
for their inner loops. iOS also forbids process forking, and Python's default
execution model assumes it.

**A phone is not a trading server.** Even if the stack loaded, the platform
needs to run a long-lived event loop, hold a database, bind a socket, and
reconcile positions against a broker. Mobile OSes suspend background
processes and confine the filesystem.

If you were told otherwise — by a tool, a template, or an AI — the claim was
wrong, and the failure would surface at install time as an `ImportError` on
`numpy`.

The honest mobile answer is a **client**: the platform runs on a server, and the
phone talks to it. That is what `quantforge.api.mobile` is.

---

## Windows

Supported, with one real degradation that is reported rather than hidden.

### No POSIX resource limits

`resource.setrlimit` does not exist on Windows. The strategy sandbox therefore
**cannot bound CPU or memory** there.

What still applies on Windows:

- **Static source screening** — the AST check blocking network, database,
  process-spawning, and serialisation imports
- **Wall-clock timeout** — a runaway strategy is abandoned after 60s

```python
StrategySandbox().describe()
# {"host": "win32",
#  "resource_limits_supported": False,
#  "wall_clock_timeout_supported": True,
#  "degraded_note": "This host has no POSIX resource limits, so CPU and memory
#                   are NOT bounded by the OS here. ... Use container
#                   isolation if the memory bound matters."}
```

The sandbox is defence in depth, never a security boundary — see
[SECURITY.md](SECURITY.md). On Windows it is *less* defence in depth, and the
code says so instead of implying parity.

### File mode bits are not enforced

The live-trading authorisation is written mode 0400 on POSIX. Windows largely
ignores those bits, so the file remains writable by its owner.

**The protection is weaker, not absent in effect:** the HMAC signature still
*detects* tampering on the next gate check. What is missing is *prevention*:

```python
auth.save(path)
auth.last_write_protection
# {"protected": False, "detail": "file permission bits are not enforced ... 
#   tampering is DETECTED on the next check, but it is not PREVENTED. Run the
#   platform on Linux or macOS for filesystem-level protection."}
```

If you are enabling live trading, run it on Linux or macOS.

### What works unchanged on Windows

Everything else: the full backtest engine, the risk engine, the kill switches,
the order state machine, reconciliation, the model registry, the API. The
platform has no POSIX-only logic outside the two items above.

---

## What was actually changed

### One module decides, not scattered `sys.platform` checks

`quantforge/core/platform.py` is the only place that knows what the host can
do. Everything else asks it.

A missed `sys.platform` check is an `AttributeError` at the worst moment, and
"which OS" is the wrong question anyway — the right one is "what capability is
missing", which is what the module reports:

```python
from quantforge.core import platform as P

P.current()                       # Platform, as capabilities
P.resource_limits_supported()     # False on Windows
P.set_resource_limits(...)        # {"applied": {}, "skipped": {...}}
P.describe()                      # full report for /health and diagnostics
```

### No POSIX-only import at module scope

```python
# A top-level `import resource` breaks the whole application on Windows.
```

`quantforge/security/audit.py` used to do exactly this. It now defers the
import into a function reached only on a host that supports it, and a test
enforces the rule:

```python
def test_no_module_imports_resource_at_top_level(): ...
```

### Filesystem locations follow the OS convention

`DEFAULT_ROOT` used to be `Path.cwd() / ".quantforge"`, captured **at import
time**. That has two defects: the location moves if the process later changes
directory, and on Windows the conventional location for application data is not
the working directory.

| | Location |
|---|---|
| Linux | `$XDG_DATA_HOME/quantforge` or `~/.local/share/quantforge` |
| macOS | `~/Library/Application Support/QuantForge` |
| Windows | `%LOCALAPPDATA%\QuantForge` |

Every one is overridable: `QUANTFORGE_HOME`, `QUANTFORGE_DATA`,
`QUANTFORGE_LOGS`, `QUANTFORGE_CONFIG`, `QUANTFORGE_STATE`.

```python
from quantforge.core import platform as P
P.data_dir(); P.log_dir(); P.config_dir(); P.ensure_dirs()
```

A test asserts no machine-absolute path is hardcoded anywhere in `quantforge/`.

### Loopback by default, everywhere

`P.default_bind_host()` returns `127.0.0.1` and `APIConfig.host` defaults to
loopback. An authenticated trading API should not become network-reachable
because someone bound it to `0.0.0.0` for convenience.

---

## The mobile client

`quantforge/api/mobile.py` is **standard library only**. That is the whole
design constraint, and it is what makes the module shippable inside a mobile
app where numpy cannot load.

```python
from quantforge.api.mobile import QuantForgeClient

client = QuantForgeClient("https://quantforge.internal")
client.login("u_pm", password)

client.health()["platform"]["name"]     # what the server host is
client.submit_backtest(strategy_id="sma_cross", strategy_version="1.0.0",
                       instrument_id="AAPL")
job = client.poll_job(job_id)          # polls, does not block a request
```

It imports only `json`, `ssl`, `urllib`, `dataclasses`, `typing`. A CI job
fails the build if a third-party import appears.

Deliberate omissions:

- **No token persistence.** The token is in memory only. On a device, storing
  it is the platform keystore's job, not this module's.
- **No TLS bypass.** `verify_tls=False` exists as a localhost debugging
  affordance and is loudly reported by `describe()`.
- **No default confirmation phrase.** `live_activate()` *requires* the caller
  to pass it. A test asserts the parameter has no default.

### What it cannot do

It cannot run a backtest, train a model, or evaluate an indicator — those need
the compiled stack, so they run on the server. It submits work and reads
results.

### iOS specifically

`urllib.request` works; anything else needs a native bridge. The module is
small and stdlib-only precisely so it can be ported line by line to Swift or
Kotlin if you would rather not run Python on the device at all.

---

## Continuous integration

`.github/workflows/ci.yml` runs the full matrix:

```
ubuntu-latest / windows-latest / macos-latest  x  Python 3.11, 3.12, 3.13
```

Every job runs tests, the leakage suite, an import of every module, the
end-to-end demo, and asserts the live-trading gate is **closed**. A wheel and
sdist are built on Linux.

Windows and macOS are matrix members rather than an afterthought, because the
parts that differ between hosts are exactly the parts a Linux-only pipeline
never exercises.

A separate job proves the mobile client has no compiled dependencies.

---

## Verified vs. asserted

| Claim | How |
|---|---|
| Linux works | 280 tests, demo, live on this machine |
| Windows / macOS work | CI matrix; not run on this machine |
| No POSIX-only import | `test_no_module_imports_resource_at_top_level` |
| Limits report skips, not silence | `test_unsupported_host_reports_skips_not_silence` |
| Read-only protection is honest | `test_read_only_honesty` |
| No hardcoded machine paths | `test_no_absolute_path_from_the_build_machine_is_hardcoded` |
| Client is stdlib-only | `test_imports_standard_library_only` + a CI job |
| Mobile is not a host | `MOBILE_CLIENT`, surfaced on `/health` |

Windows and macOS support is **CI-verified, not locally verified** — I have one
machine. Run `./build.sh` (or `build.bat`) on yours.

---

## Known gaps

- No Windows-specific file locking. The audit log appends with `open(..., "a")`,
  which is atomic enough for a single writer; concurrent writers on one file
  are not coordinated.
- `case_sensitive_paths` is reported, not worked around. Code that compares
  instrument symbols should normalise case if it must run on macOS.
- No path-length handling for Windows `MAX_PATH` without long-path opt-in. Deep
  dataset trees can exceed 260 characters.
- The mobile client has no WebSocket support; it polls. Fine for a queued job,
  wrong for a streaming signal feed.
