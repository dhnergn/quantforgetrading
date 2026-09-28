"""Cross-platform compatibility tests."""
import os
import platform
import sys
from pathlib import Path

import pytest


class TestPlatformDetection:
    """Test platform detection and path handling."""

    def test_get_platform(self):
        from quantforge.core.platform import get_platform
        result = get_platform()
        assert result in ("linux", "windows", "darwin")

    def test_get_app_data_dir(self):
        from quantforge.core.platform import get_app_data_dir
        result = get_app_data_dir()
        assert isinstance(result, Path)
        assert "quantforge" in str(result).lower()

    def test_get_config_path(self):
        from quantforge.core.platform import get_config_path
        result = get_config_path()
        assert isinstance(result, Path)

    def test_get_data_dir(self):
        from quantforge.core.platform import get_data_dir
        result = get_data_dir()
        assert isinstance(result, Path)
        assert result.name == "data"

    def test_get_logs_dir(self):
        from quantforge.core.platform import get_logs_dir
        result = get_logs_dir()
        assert isinstance(result, Path)
        assert result.name == "logs"

    def test_ensure_dirs(self, tmp_path):
        from quantforge.core.platform import ensure_dirs
        ensure_dirs()

    def test_is_mobile(self):
        from quantforge.core.platform import is_mobile
        result = is_mobile()
        assert isinstance(result, bool)

    def test_get_optimal_workers(self):
        from quantforge.core.platform import get_optimal_workers
        result = get_optimal_workers()
        assert isinstance(result, int)
        assert result >= 1

    def test_get_temp_dir(self):
        from quantforge.core.platform import get_temp_dir
        result = get_temp_dir()
        assert isinstance(result, Path)

    def test_get_platform_config(self):
        from quantforge.core.platform import get_platform_config
        result = get_platform_config()
        assert isinstance(result, dict)


class TestCrossPlatformImports:
    """Test that all modules import correctly across platforms.

    These point at the canonical modules: ``quantforge.quant.indicators``
    (22 self-verifying indicators) and ``quantforge.strategy.builder``
    (versioned strategies). An earlier duplicate pair of packages,
    ``quantforge.indicators`` and ``quantforge.strategies``, was removed; see
    ``_legacy_scaffold/README.md``. Those had no capability the canonical
    modules lack, and their ``RSI.calculate()`` returned the whole input frame
    with the OHLC columns overwritten by the indicator value.
    """

    def test_indicators_import(self):
        from quantforge.quant.indicators import (
            ATR, EMA, MACD, RSI, SMA, BollingerBands,
        )
        assert SMA is not None
        # The canonical set is self-verifying; the duplicate was not.
        from quantforge.quant.indicators import validate_metadata
        assert validate_metadata() == []

    def test_strategies_import(self):
        from quantforge.strategy.builder import (
            bollinger_mean_reversion_strategy, rsi_mean_reversion_strategy,
            sma_crossover_strategy,
        )
        assert sma_crossover_strategy(20, 50).strategy_id == "sma_cross"
        assert rsi_mean_reversion_strategy() is not None
        assert bollinger_mean_reversion_strategy() is not None

    def test_duplicate_module_paths_are_gone(self):
        """One implementation per concern, so the two cannot drift.

        Two packages that both answered `RSI` produced two `calculate()`
        contracts, and the malformed one imported fine.
        """
        import importlib.util as u

        for shadow in ("quantforge.indicators", "quantforge.strategies"):
            assert u.find_spec(shadow) is None, (
                f"{shadow} reappeared and would shadow the canonical module"
            )

    def test_risk_import(self):
        from quantforge.risk.limits import RiskLimits
        from quantforge.risk.kill_switch import KillSwitch
        assert RiskLimits is not None
        assert KillSwitch is not None

    def test_core_import(self):
        from quantforge.core.platform import get_platform, get_app_data_dir
        assert get_platform is not None
        assert get_app_data_dir is not None


class TestPackaging:
    """A stale PACKAGES entry fails the install, so it must fail the build."""

    def test_every_declared_package_exists(self):
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "setup.py").read_text())
        packages = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and getattr(
                node.targets[0], "id", ""
            ) == "PACKAGES":
                packages = [e.value for e in node.value.elts]
                break
        assert packages, "setup.py declares no PACKAGES"
        missing = [
            p for p in packages
            if not (root / p.replace(".", "/") / "__init__.py").exists()
        ]
        assert not missing, f"PACKAGES lists non-existent packages: {missing}"

    def test_every_real_package_is_declared(self):
        """The inverse: a real package missing from PACKAGES ships nowhere."""
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "setup.py").read_text())
        declared = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and getattr(
                node.targets[0], "id", ""
            ) == "PACKAGES":
                declared = {e.value for e in node.value.elts}
                break
        pkg_root = root / "quantforge"
        actual = {"quantforge"}
        for child in pkg_root.iterdir():
            if child.is_dir() and (child / "__init__.py").exists():
                actual.add(f"quantforge.{child.name}")
        assert not (actual - declared), (
            f"real packages absent from setup.py PACKAGES: {sorted(actual - declared)}"
        )


class TestPWAFiles:
    """Test PWA files exist for mobile compatibility."""

    def test_manifest_exists(self):
        manifest = Path(__file__).parent.parent / "quantforge" / "api" / "static" / "manifest.json"
        assert manifest.exists()

    def test_service_worker_exists(self):
        sw = Path(__file__).parent.parent / "quantforge" / "api" / "static" / "sw.js"
        assert sw.exists()

    def test_index_html_exists(self):
        index = Path(__file__).parent.parent / "quantforge" / "api" / "static" / "index.html"
        assert index.exists()

    def test_icons_exist(self):
        icon_192 = Path(__file__).parent.parent / "quantforge" / "api" / "static" / "icon-192.png"
        icon_512 = Path(__file__).parent.parent / "quantforge" / "api" / "static" / "icon-512.png"
        assert icon_192.exists()
        assert icon_512.exists()


class TestDockerFiles:
    """Test Docker files exist for containerized deployment."""

    def test_dockerfile_exists(self):
        dockerfile = Path(__file__).parent.parent / "Dockerfile"
        assert dockerfile.exists()

    def test_docker_compose_exists(self):
        compose = Path(__file__).parent.parent / "docker-compose.yml"
        assert compose.exists()


class TestBuildScripts:
    """Test build scripts exist for Linux and Windows."""

    def test_build_sh_exists(self):
        build_sh = Path(__file__).parent.parent / "build.sh"
        assert build_sh.exists()

    def test_build_bat_exists(self):
        build_bat = Path(__file__).parent.parent / "build.bat"
        assert build_bat.exists()
