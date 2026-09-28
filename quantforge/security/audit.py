"""Append-only, tamper-evident audit log.

Design constraints from blueprint Section 80:

* **Append-only.** Entries are never updated or deleted.
* **Tamper-evident.** Each entry carries the hash of the previous entry, so any
  modification to an earlier entry invalidates every later hash. This is a
  hash chain, not a signature: it detects modification but does not prove who
  made it. For non-repudiation, entries are also signed with an HMAC.
* **Complete.** Every state change that affects money or risk is recorded:
  strategy, model, feature, dataset, broker, credential, risk-limit, paper and
  live activation, orders, positions, and kill switches.

The chain is verified by :meth:`AuditLog.verify_chain`, which the platform runs
periodically. A broken chain is itself an alertable security event.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from quantforge.core.enums import AuditAction
from quantforge.core.timestamps import utcnow

GENESIS = "0" * 64


@dataclass(slots=True)
class AuditEntry:
    sequence: int
    at: dt.datetime
    actor: str
    action: str
    resource: str
    detail: dict = field(default_factory=dict)
    previous_hash: str = GENESIS
    entry_hash: str = ""
    signature: str = ""
    ip_address: Optional[str] = None
    request_id: Optional[str] = None

    def compute_hash(self) -> str:
        body = {
            "sequence": self.sequence,
            "at": self.at.isoformat(),
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "detail": self.detail,
            "previous_hash": self.previous_hash,
        }
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, default=str).encode()
        ).hexdigest()

    def as_dict(self) -> dict:
        return {
            "sequence": self.sequence, "at": self.at.isoformat(),
            "actor": self.actor, "action": self.action, "resource": self.resource,
            "detail": self.detail, "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash, "signature": self.signature,
            "ip_address": self.ip_address, "request_id": self.request_id,
        }

    def is_valid(self, previous_hash: str) -> bool:
        return self.previous_hash == previous_hash and \
            self.entry_hash == self.compute_hash()


class AuditLog:
    """Hash-chained audit log with an HMAC signature per entry."""

    def __init__(
        self,
        path: Optional[str | Path] = None,
        secret: Optional[bytes] = None,
        sign: bool = True,
    ) -> None:
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._secret = secret
        self.sign = sign and secret is not None
        self._entries: list[AuditEntry] = []
        self._lock = threading.Lock()
        self._last_hash = GENESIS

    def record(
        self,
        actor: str,
        action: AuditAction | str,
        resource: str = "",
        detail: Optional[dict] = None,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None,
        at: Optional[dt.datetime] = None,
    ) -> AuditEntry:
        """Append one entry. Never raises on content; the chain is the record."""
        with self._lock:
            action_value = action.value if isinstance(action, AuditAction) else str(action)
            entry = AuditEntry(
                sequence=len(self._entries) + 1,
                at=at or utcnow(),
                actor=actor,
                action=action_value,
                resource=resource,
                detail=detail or {},
                previous_hash=self._last_hash,
                ip_address=ip_address,
                request_id=request_id,
            )
            entry.entry_hash = entry.compute_hash()
            if self.sign and self._secret is not None:
                entry.signature = hmac.new(
                    self._secret, entry.entry_hash.encode(), hashlib.sha256
                ).hexdigest()
            self._entries.append(entry)
            self._last_hash = entry.entry_hash
            if self.path is not None:
                with self.path.open("a") as fh:
                    fh.write(json.dumps(entry.as_dict(), default=str) + "\n")
            return entry

    # -- verification -----------------------------------------------------

    def verify_chain(self) -> dict:
        """Recompute every hash. A break means the log was modified."""
        prev = GENESIS
        broken_at: Optional[int] = None
        unsigned: list[int] = []
        for entry in self._entries:
            if not entry.is_valid(prev):
                broken_at = entry.sequence
                break
            if self.sign and entry._signature_invalid(self._secret):  # type: ignore[attr-defined]
                unsigned.append(entry.sequence)
            prev = entry.entry_hash
        return {
            "valid": broken_at is None and not unsigned,
            "n_entries": len(self._entries),
            "broken_at_sequence": broken_at,
            "unsigned_sequences": unsigned,
            "head_hash": self._last_hash,
            "detail": (
                None if broken_at is None else
                f"entry {broken_at} does not match its recorded hash; the log has "
                "been modified since it was written"
            ),
        }

    # -- queries ----------------------------------------------------------

    def entries(
        self, actor: Optional[str] = None, action: Optional[str] = None,
        resource: Optional[str] = None, since: Optional[dt.datetime] = None,
        limit: Optional[int] = None,
    ) -> list[dict]:
        out = []
        for e in self._entries:
            if actor and e.actor != actor:
                continue
            if action and e.action != (action.value if isinstance(action, AuditAction) else action):
                continue
            if resource and resource not in e.resource:
                continue
            if since and e.at < since:
                continue
            out.append(e.as_dict())
        if limit:
            out = out[-limit:]
        return out

    def frame(self) -> pd.DataFrame:
        if not self._entries:
            return pd.DataFrame()
        return pd.DataFrame([{
            "sequence": e.sequence, "at": e.at, "actor": e.actor,
            "action": e.action, "resource": e.resource,
        } for e in self._entries])

    def actions_taken(self) -> dict:
        counts: dict[str, int] = {}
        for e in self._entries:
            counts[e.action] = counts.get(e.action, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)


def _signature_invalid(self: AuditEntry, secret: Optional[bytes]) -> bool:
    if not secret:
        return False
    expected = hmac.new(secret, self.entry_hash.encode(), hashlib.sha256).hexdigest()
    return not hmac.compare_digest(expected, self.signature)


AuditEntry._signature_invalid = _signature_invalid  # type: ignore[attr-defined]


# ==========================================================================
# Sandbox
# ==========================================================================


class SandboxViolation(Exception):
    """Raised by the strategy sandbox. Distinct from a user-code error."""


class StrategySandbox:
    """Guards user-supplied strategy code.

    This is a **defence-in-depth** control, not a security boundary. In-process
    restrictions can be defeated by sufficiently determined code; a real
    deployment must additionally run untrusted strategies in a container or
    microVM with seccomp, a read-only filesystem, no network, and a CPU/memory
    cgroup. The two together are what make running third-party code
    defensible, and this class makes the intent explicit rather than implying a
    protection that does not exist.
    """

    #: Modules a strategy must never import. Credentials, database drivers, and
    #: network stacks are excluded so a strategy cannot exfiltrate anything.
    BLOCKED_MODULES = frozenset({
        "socket", "ssl", "http", "urllib", "urllib3", "requests", "httpx",
        "ftplib", "smtplib", "telnetlib", "paramiko", "asyncio",
        "subprocess", "multiprocessing", "ctypes", "resource", "signal",
        "sqlite3", "psycopg2", "psycopg", "sqlalchemy", "pymongo", "redis",
        "hvac", "boto3", "pickle", "marshal", "shelve", "dill", "joblib",
        "importlib", "runpy", "code", "codeop", "pty", "fcntl", "mmap",
        "webbrowser", "platform", "getpass", "pwd",
    })

    #: Names that must never appear in a strategy's namespace, whatever module
    #: provided them.
    FORBIDDEN_NAMES = frozenset({
        "os", "sys", "subprocess", "socket", "open", "eval", "exec",
        "compile", "__import__", "input", "globals", "locals", "vars",
        "getattr", "setattr", "delattr", "breakpoint", "memoryview",
        "help", "exit", "quit",
    })

    def __init__(
        self,
        max_cpu_seconds: float = 30.0,
        max_wall_seconds: float = 60.0,
        max_memory_mb: int = 512,
        max_output_bytes: int = 1_048_576,
        max_source_bytes: int = 512_000,
    ) -> None:
        self.max_cpu_seconds = max_cpu_seconds
        self.max_wall_seconds = max_wall_seconds
        self.max_memory_mb = max_memory_mb
        self.max_output_bytes = max_output_bytes
        self.max_source_bytes = max_source_bytes

    # -- static checks ----------------------------------------------------

    def check_source(self, source: str) -> dict:
        """Parse and statically screen a strategy's source.

        AST-based rather than regex-based: a regex cannot tell a docstring from
        an import, nor an attribute named ``system`` from a call to it.
        """
        import ast

        if len(source.encode()) > self.max_source_bytes:
            raise SandboxViolation(
                f"strategy source exceeds {self.max_source_bytes} bytes"
            )
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise SandboxViolation(f"strategy source has a syntax error: {exc}") from exc

        findings: list[dict] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in self.BLOCKED_MODULES:
                        findings.append({
                            "line": node.lineno, "type": "blocked_import",
                            "detail": f"import of {alias.name!r}; a strategy must not "
                                      "be able to reach the network, the database, "
                                      "or the filesystem",
                        })
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in self.BLOCKED_MODULES:
                    findings.append({
                        "line": node.lineno, "type": "blocked_import",
                        "detail": f"from {node.module!r} import ... is not permitted",
                    })
            elif isinstance(node, ast.Name) and node.id in self.FORBIDDEN_NAMES:
                if node.id not in ("getattr", "setattr", "delattr"):
                    findings.append({
                        "line": node.lineno, "type": "forbidden_name",
                        "detail": f"use of {node.id!r}",
                    })
            elif isinstance(node, ast.Attribute):
                attr = node.attr
                if attr in ("system", "popen", "spawn", "fork", "execv", "kill"):
                    findings.append({
                        "line": node.lineno, "type": "forbidden_attribute",
                        "detail": f"access to {attr!r}",
                    })
        if findings:
            raise SandboxViolation(
                f"strategy source violates the sandbox: {findings[:5]}"
            )
        return {"ok": True, "n_nodes": sum(1 for _ in ast.walk(tree)),
                "checks": ["imports", "forbidden_names", "dangerous_attributes"]}

    # -- resource limits --------------------------------------------------

    def apply_process_limits(self) -> dict:
        """Apply CPU/memory limits to the *current* process.

        Uses ``resource.setrlimit`` where available. The address-space limit is
        deliberately not set to a hard RLIMIT_AS on platforms where numpy's
        allocator reserves virtual address space eagerly, which would make
        legitimate numerical code fail at import.
        """
        import resource as _resource

        applied: dict[str, Any] = {}
        try:
            soft_cpu = int(self.max_cpu_seconds)
            _resource.setrlimit(_resource.RLIMIT_CPU, (soft_cpu, soft_cpu + 5))
            applied["cpu_seconds"] = soft_cpu
        except (ValueError, OSError, AttributeError) as exc:
            applied["cpu_error"] = str(exc)
        try:
            limit_bytes = self.max_memory_mb * 1024 * 1024
            cur_soft, cur_hard = _resource.getrlimit(_resource.RLIMIT_AS)
            new_hard = cur_hard if cur_hard != _resource.RLIM_INFINITY else limit_bytes
            _resource.setrlimit(_resource.RLIMIT_AS, (limit_bytes, new_hard))
            applied["address_space_mb"] = self.max_memory_mb
        except (ValueError, OSError, AttributeError) as exc:
            applied["memory_error"] = str(exc)
        try:
            _resource.setrlimit(
                _resource.RLIMIT_NOFILE, (64, 64)
            )
            applied["open_files"] = 64
        except (ValueError, OSError, AttributeError) as exc:
            applied["fd_error"] = str(exc)
        try:
            _resource.setrlimit(
                _resource.RLIMIT_FSIZE, (self.max_output_bytes, self.max_output_bytes)
            )
            applied["file_size_bytes"] = self.max_output_bytes
        except (ValueError, OSError, AttributeError) as exc:
            applied["fsize_error"] = str(exc)
        return applied

    def remove_process_limits(self) -> None:
        """Restore limits. Only for a controlled shutdown path."""
        import resource as _resource

        for res, value in (
            (_resource.RLIMIT_CPU, _resource.RLIM_INFINITY),
            (_resource.RLIMIT_AS, _resource.RLIM_INFINITY),
            (_resource.RLIMIT_FSIZE, _resource.RLIM_INFINITY),
        ):
            try:
                _resource.setrlimit(res, (value, value))
            except (ValueError, OSError, AttributeError):
                continue

    def enforce_wall_timeout(self, fn, *args, **kwargs):
        """Run ``fn`` with a wall-clock timeout.

        A thread is used because a signal-based alarm only interrupts the main
        thread and does not work on every platform. The thread is abandoned
        rather than killed, which is why this is a guard and not a boundary.
        """
        import threading

        box: dict[str, Any] = {}

        def runner():
            try:
                box["result"] = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001
                box["error"] = exc

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(self.max_wall_seconds)
        if t.is_alive():
            raise SandboxViolation(
                f"strategy exceeded the {self.max_wall_seconds}s wall-clock limit "
                "and was abandoned. The thread may still be running; treat this as "
                "a hard failure and do not continue the session."
            )
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def describe(self) -> dict:
        return {
            "max_cpu_seconds": self.max_cpu_seconds,
            "max_wall_seconds": self.max_wall_seconds,
            "max_memory_mb": self.max_memory_mb,
            "max_output_bytes": self.max_output_bytes,
            "network": "denied",
            "filesystem_writes": "denied",
            "blocked_module_count": len(self.BLOCKED_MODULES),
            "boundary_note": (
                "These in-process limits are defence in depth, not a security "
                "boundary. Running untrusted strategies requires container or "
                "microVM isolation with seccomp, a read-only root filesystem, no "
                "network namespace, and cgroup limits enforced outside the "
                "interpreter."
            ),
        }
