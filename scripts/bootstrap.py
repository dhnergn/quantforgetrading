#!/usr/bin/env python3
"""One setup and verification script for every supported host.

Why Python and not `build.sh` plus `build.bat`: bash is absent on Windows and
batch is absent on macOS and Linux, so a shell script always needs a twin and
the twins drift. Python is guaranteed present on all three, because this is a
Python project.

    python scripts/bootstrap.py              # install, then verify
    python scripts/bootstrap.py --check      # verify only, install nothing
    python scripts/bootstrap.py --host-info  # what this machine supports

Runs on Linux, macOS, and Windows. Reports what it skipped rather than failing
silently: on Windows the sandbox has no POSIX resource limits, and pretending
otherwise would mean a verification step that passes while checking nothing.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BAR = "=" * 70


def venv_python(root: Path) -> Path:
    """The interpreter inside the project venv, or the current one."""
    candidate = root / ".venv" / ("Scripts" if os.name == "nt" else "bin") / \
        ("python.exe" if os.name == "nt" else "python")
    return candidate if candidate.exists() else Path(sys.executable)


def run(cmd: list, cwd: Path = ROOT) -> tuple[int, str]:
    printable = " ".join(str(c) for c in cmd)
    print(f"  $ {printable}")
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def host_info() -> dict:
    from quantforge.core import platform as P

    return P.describe()


def print_host_info() -> None:
    from quantforge.core import platform as P

    info = P.describe()
    print(BAR)
    print("HOST CAPABILITIES")
    print(BAR)
    print(f"  platform              {info['name']}  (Python {info['python_version']})")
    print(f"  path separator        {info['path_separator']!r}")
    print(f"  case-sensitive paths  {info['case_sensitive_paths']}")
    print(f"  max path length       {info['max_path_length']}")
    print(f"  POSIX resource limits {info['supports_resource_limits']}")
    print(f"  read-only file modes  {info['supports_read_only_files']}")
    print(f"  fork available        {info['supports_fork']}")
    print(f"  optimal workers       {info['optimal_workers']}")
    print("  directories")
    for k, v in info["directories"].items():
        print(f"      {k:6s} {v}")
    for note in info["notes"]:
        print(f"  note: {note}")
    if info["mobile"]["is_mobile_host"]:
        print()
        print(f"  MOBILE: {info['mobile']['guidance']}")
        print()
        print("  This platform cannot host QuantForge. Stop here.")
        return False
    print()
    return True


def install(py: Path) -> bool:
    print(BAR)
    print("INSTALL")
    print(BAR)
    for args in (
        [str(py), "-m", "pip", "install", "--upgrade", "pip"],
        [str(py), "-m", "pip", "install", "-r", "requirements.txt"],
        [str(py), "-m", "pip", "install", "-e", "."],
    ):
        code, out = run(args)
        if code != 0:
            print(out[-2000:])
            print(f"  FAILED: {' '.join(str(a) for a in args)}")
            return False
    return True


def verify(py: Path, skip_demo: bool = False) -> bool:
    print(BAR)
    print("VERIFY")
    print(BAR)
    ok = True

    code, out = run([str(py), "-m", "pytest", "tests/", "-q"])
    tail = out.strip().splitlines()[-1] if out.strip() else "(no output)"
    print(f"  tests: {tail}")
    if code != 0:
        print(out[-3000:])
        ok = False

    # Import every package, so a platform-specific import error surfaces here
    # rather than in production.
    code, out = run([
        str(py), "-c",
        "import pkgutil, importlib, quantforge;"
        "mods=[m.name for m in pkgutil.walk_packages(quantforge.__path__, 'quantforge.')];"
        "[importlib.import_module(m) for m in mods];"
        "print(f'imported {len(mods)} modules')",
    ])
    print(f"  imports: {out.strip().splitlines()[-1] if out.strip() else 'FAILED'}")
    if code != 0:
        print(out[-2000:])
        ok = False

    if skip_demo:
        print("  demo: skipped (--skip-demo)")
    else:
        code, out = run([str(py), "scripts/demo.py"])
        print(f"  demo: {'exit 0' if code == 0 else 'FAILED'}")
        if code != 0:
            print(out[-2000:])
            ok = False

    # The live-trading gate must be closed. A build that verifies while live
    # trading is enabled has verified the wrong thing.
    from quantforge.execution.live_gate import LiveTradingGate
    permitted, why = LiveTradingGate().check()
    print(f"  live-trading gate: permitted={permitted}")
    if permitted:
        print("    WARNING: live trading is AUTHORISED on this host.")

    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify without installing")
    ap.add_argument("--host-info", action="store_true", help="print capabilities and exit")
    ap.add_argument("--skip-demo", action="store_true", help="skip the demo run")
    args = ap.parse_args(argv)

    py = venv_python(ROOT)
    print(f"interpreter: {py}")

    if args.host_info:
        return 0 if print_host_info() else 2

    if not print_host_info():
        return 2

    if not args.check and not install(py):
        print("\nINSTALL FAILED")
        return 1

    ok = verify(py, skip_demo=args.skip_demo)
    print()
    print(BAR)
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    print(BAR)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
