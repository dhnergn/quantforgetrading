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
    """Test that all modules import correctly across platforms."""

    def test_indicators_import(self):
        from quantforge.indicators import SMA, EMA, RSI, MACD, BollingerBands, ATR
        assert SMA is not None

    def test_strategies_import(self):
        from quantforge.strategies import CrossoverStrategy, MeanReversionStrategy, BreakoutStrategy
        assert CrossoverStrategy is not None

    def test_risk_import(self):
        from quantforge.risk.limits import RiskLimits
        from quantforge.risk.kill_switch import KillSwitch
        assert RiskLimits is not None

    def test_core_import(self):
        from quantforge.core.platform import get_platform, get_app_data_dir
        assert get_platform is not None


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
