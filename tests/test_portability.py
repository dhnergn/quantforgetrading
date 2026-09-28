"""Cross-platform support: what the code actually does on each host.

A file-existence check is not a cross-platform test. These assert the
properties that decide whether the platform works on a given OS:

* the POSIX-only ``resource`` module is never imported at module scope, because
  on Windows ``import resource`` raises ``ModuleNotFoundError`` and takes the
  whole application with it;
* resource limits that a host cannot apply are **reported as skipped** rather
  than silently absent;
* filesystem locations follow the OS convention and honour their env override;
* read-only protection states honestly whether the host enforces it;
* the mobile client imports nothing outside the standard library, which is
  what makes it shippable in a phone app.

The existing file-existence and platform-detection checks live in
``test_cross_platform.py``; this file is the behavioural half.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from quantforge.core import platform as P

ROOT = Path(__file__).resolve().parents[1]
IS_WINDOWS = os.name == "nt"
IS_POSIX = os.name == "posix"


# ==========================================================================
# The module must import everywhere
# ==========================================================================


class TestPortableImport:
    def test_platform_module_imports_without_resource(self):
        """`resource` is POSIX-only; importing it at module scope breaks Windows."""
        assert P.current().name in ("linux", "darwin", "win32", "cygwin")

    def test_no_module_imports_resource_at_top_level(self):
        """A top-level `import resource` in any module breaks Windows entirely.

        The import must be deferred inside a function that is only reached when
        the host supports it.
        """
        offenders = []
        for path in (ROOT / "quantforge").rglob("*.py"):
            src = path.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(src.splitlines(), 1):
                stripped = line.strip()
                if not stripped.startswith(("import resource", "from resource")):
                    continue
                # Deferred imports are indented; a top-level one is not.
                if not line.startswith((" ", "\t")):
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno}")
        assert not offenders, (
            "top-level `import resource` breaks the whole application on "
            f"Windows: {offenders}"
        )

    def test_every_package_imports(self):
        """Import every module in a subprocess so one failure names itself."""
        code = (
            "import pkgutil, importlib, quantforge;"
            "mods=[m.name for m in pkgutil.walk_packages("
            "quantforge.__path__, 'quantforge.')];"
            "[importlib.import_module(m) for m in mods];"
            "print(len(mods))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT,
            capture_output=True, text=True, timeout=180,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert int(proc.stdout.strip()) > 30


# ==========================================================================
# Resource limits
# ==========================================================================


class TestResourceLimits:
    def test_reports_support_honestly(self):
        supported = P.resource_limits_supported()
        assert supported == (IS_POSIX and not IS_WINDOWS)
        # The flag is what callers branch on, so it must not be a guess.

    def test_unsupported_host_reports_skips_not_silence(self):
        report = P.set_resource_limits(
            max_cpu_seconds=30, max_address_space_mb=512,
            max_open_files=64, max_file_size_mb=1,
        )
        assert set(report) >= {"applied", "skipped", "supported", "note"}
        if not report["supported"]:
            # Every requested limit must be accounted for by name.
            for key in ("cpu_seconds", "address_space_mb", "open_files", "file_size_mb"):
                assert key in report["skipped"], f"{key} vanished instead of being reported"
            assert report["note"], "an unsupported host must explain itself"
        else:
            assert report["applied"].get("cpu_seconds") == 30

    def test_clearing_limits_is_safe_on_every_host(self):
        """Must not raise on a host with no limits to clear."""
        P.clear_resource_limits()
        P.clear_resource_limits()

    def test_sandbox_describe_matches_the_host(self):
        from quantforge.security.audit import StrategySandbox

        d = StrategySandbox().describe()
        assert d["resource_limits_supported"] == P.resource_limits_supported()
        assert d["wall_clock_timeout_supported"] is True
        assert "not a security boundary" in d["boundary_note"]
        if not d["resource_limits_supported"]:
            assert "degraded_note" in d, (
                "a host without resource limits must say the sandbox is weaker"
            )


# ==========================================================================
# Filesystem locations
# ==========================================================================


class TestDirectories:
    @pytest.mark.parametrize("fn,leaf", [
        (P.data_dir, "data"),
        (P.log_dir, "logs"),
    ])
    def test_conventional_locations(self, fn, leaf):
        p = fn()
        assert p.name == leaf
        assert p.is_absolute()

    def test_env_override_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTFORGE_DATA", str(tmp_path))
        assert P.data_dir() == tmp_path

    def test_app_dir_respects_its_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTFORGE_HOME", str(tmp_path))
        assert P.app_dir() == tmp_path

    def test_no_absolute_path_from_the_build_machine_is_hardcoded(self):
        """A path baked at build time breaks every other machine.

        This is the defect that had `DEFAULT_ROOT` capture ``Path.cwd()`` at
        import time, which moved whenever the process changed directory.
        """
        offenders = []
        for path in (ROOT / "quantforge").rglob("*.py"):
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                for bad in ("/home/", "C:\\\\Users\\\\", "/Users/"):
                    if bad in line and "http" not in line:
                        offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {bad}")
        assert not offenders, f"hardcoded machine paths: {offenders}"

    def test_ensure_dirs_creates_and_reports(self, tmp_path, monkeypatch):
        for var in ("QUANTFORGE_DATA", "QUANTFORGE_LOGS", "QUANTFORGE_CONFIG",
                    "QUANTFORGE_STATE"):
            monkeypatch.setenv(var, str(tmp_path / var))
        created = P.ensure_dirs()
        assert set(created) == {"data", "logs", "config", "state"}
        for label, value in created.items():
            assert not str(value).startswith("FAILED"), f"{label}: {value}"
            assert Path(value).is_dir()

    def test_optimal_workers_is_at_least_one(self):
        assert P.get_optimal_workers() >= 1

    def test_temp_dir_is_writable(self):
        probe = P.temp_dir() / ".qf_probe"
        try:
            probe.write_text("x")
            assert probe.exists()
        finally:
            probe.unlink(missing_ok=True)


# ==========================================================================
# File permissions
# ==========================================================================


class TestFilePermissions:
    def test_read_only_honesty(self, tmp_path):
        f = tmp_path / "auth.json"
        f.write_text("{}")
        result = P.make_read_only(f)
        if P.current().supports_read_only_files:
            assert result["protected"] is True
            assert P.is_read_only(f) is True
        else:
            # The important half: it must NOT claim protection it lacks.
            assert result["protected"] is False
            assert "not enforced" in result["detail"]
            assert P.is_read_only(f) is False

    def test_live_gate_reports_the_protection_it_got(self, tmp_path):
        from quantforge.execution.live_gate import LiveAuthorisation, build_authorisation

        auth = build_authorisation("tester", confirmed=[])
        auth.sign(b"k" * 40)
        path = auth.save(tmp_path / "auth.json")
        protection = auth.last_write_protection
        assert "protected" in protection
        assert protection["protected"] == P.current().supports_read_only_files

    def test_authorisation_still_loads_after_save(self, tmp_path):
        from quantforge.execution.live_gate import LiveAuthorisation, build_authorisation

        auth = build_authorisation("tester", confirmed=[])
        auth.sign(b"k" * 40)
        path = auth.save(tmp_path / "auth.json")
        loaded = LiveAuthorisation.load(path)
        assert loaded is not None and loaded.authorised_by == "tester"
        # A read-only file must not become unreadable.
        assert loaded.verify(b"k" * 40)


# ==========================================================================
# Networking
# ==========================================================================


class TestNetworkDefaults:
    def test_binds_to_loopback_by_default(self):
        """An authenticated trading API should not be network-reachable by default."""
        assert P.default_bind_host() in ("127.0.0.1", "::1")

    def test_api_defaults_to_loopback(self):
        from quantforge.core.config import APIConfig

        assert APIConfig().host in ("127.0.0.1", "::1", "localhost")


# ==========================================================================
# Mobile client
# ==========================================================================


class TestMobileClient:
    def test_imports_standard_library_only(self):
        """The whole point: it must load where numpy/pandas cannot."""
        import ast

        tree = ast.parse((ROOT / "quantforge" / "api" / "mobile.py").read_text())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
        third_party = {
            m for m in mods
            if m not in sys.stdlib_module_names and not m.startswith("quantforge")
        }
        assert not third_party, (
            f"the mobile client must have no compiled dependencies, found: "
            f"{sorted(third_party)}"
        )

    def test_no_tls_verification_bypass_in_source(self):
        """Check the AST, not the text.

        A text search matches the word inside a docstring that *describes* the
        prohibition, which is how this test first failed - against its own
        module, whose docstring says "no ``verify=False``".
        """
        import ast

        tree = ast.parse((ROOT / "quantforge" / "api" / "mobile.py").read_text())
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if (
                        kw.arg == "verify"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is False
                    ):
                        offenders.append(f"line {node.lineno}: verify=False")
            if isinstance(node, ast.Attribute) and node.attr == "_create_unverified_context":
                offenders.append(f"line {node.lineno}: _create_unverified_context")
        assert not offenders, f"TLS bypass in executable code: {offenders}"

    def test_insecure_tls_is_reachable_only_explicitly(self):
        """`verify_tls=False` must be opt-in and visibly warned about.

        A client that silently accepted any certificate would expose a bearer
        token to interception, so the insecure path has to leave a trace in
        ``describe()``.
        """
        from quantforge.api.mobile import QuantForgeClient

        assert QuantForgeClient("https://x.invalid").describe()["warning"] is None
        assert QuantForgeClient(
            "https://x.invalid", verify_tls=False
        ).describe()["warning"] is not None

    def test_rejects_non_http_urls(self):
        from quantforge.api.mobile import QuantForgeClient

        for bad in ("ftp://host", "file:///etc/passwd", "", "not a url"):
            with pytest.raises(ValueError):
                QuantForgeClient(bad)

    def test_does_not_persist_the_token(self):
        src = (ROOT / "quantforge" / "api" / "mobile.py").read_text()
        for banned in ("open(token", "json.dump(self._token", "keyring"):
            assert banned not in src

    def test_describe_reports_insecure_tls(self):
        from quantforge.api.mobile import QuantForgeClient

        c = QuantForgeClient("https://example.invalid", verify_tls=False)
        d = c.describe()
        assert d["tls_verified"] is False
        assert d["warning"] and "DISABLED" in d["warning"]
        # And it must not leak the token, which it does not hold here anyway.
        assert "token" not in d or d["token"] is None

    def test_live_activate_requires_a_passed_confirmation(self):
        """The client must not construct or default the confirmation phrase."""
        import inspect

        from quantforge.api.mobile import QuantForgeClient

        sig = inspect.signature(QuantForgeClient.live_activate)
        assert "confirmation" in sig.parameters
        default = sig.parameters["confirmation"].default
        assert default is inspect.Parameter.empty, (
            "the confirmation phrase must be required, not defaulted"
        )
        src = (ROOT / "quantforge" / "api" / "mobile.py").read_text()
        assert "I UNDERSTAND THIS PLACES REAL ORDERS" not in src


# ==========================================================================
# Mobile hosts
# ==========================================================================


class TestMobileHostGuidance:
    def test_mobile_guidance_is_actionable(self):
        from quantforge.core.platform import MOBILE_CLIENT

        assert "not supported as compute hosts" in MOBILE_CLIENT
        # It must say what to do instead, not just refuse.
        assert "quantforge.api.mobile" in MOBILE_CLIENT

    def test_describe_lists_supported_hosts(self):
        info = P.describe()
        assert info["supported_hosts"] == ["linux", "darwin", "win32"]
        assert "guidance" in info["mobile"]

    def test_is_mobile_returns_bool_on_any_host(self):
        assert isinstance(P.is_mobile(), bool)


# ==========================================================================
# Cross-process file locking
# ==========================================================================


class TestFileLocking:
    """The audit trail must survive two processes writing it at once.

    A ``threading.Lock`` only covers one process. Running the API and a worker
    as separate processes is ordinary, and two writers interleaving half a JSON
    line produces a corrupt audit trail - the one record that has to be
    trustworthy. These tests spawn real processes rather than threads, because
    threads cannot exercise the failure they exist to prevent.
    """

    def test_backend_is_reported(self):
        assert P.file_lock_backend() in ("fcntl", "msvcrt", "none")

    def test_lock_is_reentrant_across_sequential_uses(self, tmp_path):
        target = tmp_path / "x.log"
        for _ in range(3):
            with P.file_lock(target) as r:
                assert r["locked"] is True or r["backend"] == "none"

    def test_releases_the_lock_afterwards(self, tmp_path):
        target = tmp_path / "x.log"
        with P.file_lock(target):
            pass
        # A second acquisition must not block, which is the only externally
        # visible sign the first released.
        with P.file_lock(target, timeout=0.5) as r:
            assert r["locked"] is True or r["backend"] == "none"

    def test_two_processes_do_not_interleave(self, tmp_path):
        """The real test: N processes appending to one audit file."""
        import subprocess
        import textwrap

        log = tmp_path / "audit.jsonl"
        script = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(ROOT)!r})
            from pathlib import Path
            from quantforge.core.platform import file_lock
            target = Path({str(log)!r})
            payload = "x" * 400 + "\\n"
            for _ in range(40):
                with file_lock(target, timeout=30.0):
                    with target.open("a") as fh:
                        fh.write(payload)
        """)
        script_path = tmp_path / "writer.py"
        script_path.write_text(script)

        procs = [
            subprocess.Popen([sys.executable, str(script_path)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for _ in range(4)
        ]
        for p in procs:
            out, err = p.communicate(timeout=180)
            assert p.returncode == 0, f"writer failed: {err.decode()[-1500:]}"

        lines = log.read_text().splitlines()
        assert len(lines) == 160, f"expected 160 lines, got {len(lines)}"
        # Interleaving would show up as a short or corrupted line.
        assert all(len(line) == 400 for line in lines), "a line was interleaved"

    def test_audit_log_uses_the_lock(self, tmp_path):
        """The writers themselves must take the lock, not merely offer it."""
        from quantforge.monitoring.observability import ExecutionLogger
        from quantforge.security.audit import AuditLog

        audit = AuditLog(path=tmp_path / "a.jsonl", secret=b"k" * 40)
        audit.record("alice", "user_login")
        assert (tmp_path / "a.jsonl").read_text().strip()
        assert (tmp_path / "a.jsonl.lock").exists(), "no lock file was created"

        ex = ExecutionLogger(path=tmp_path / "e.jsonl")
        ex.log("o1", "signal")
        assert (tmp_path / "e.jsonl.lock").exists()

    def test_lock_degrades_honestly(self, tmp_path, monkeypatch):
        """A host with no primitive must not claim cross-process protection."""
        monkeypatch.setattr(P, "file_lock_backend", lambda: "none")
        with P.file_lock(tmp_path / "y.log") as r:
            assert r["locked"] is False
            assert r["backend"] == "none"
            assert "does NOT protect" in r["detail"]
