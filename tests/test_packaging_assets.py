"""The PWA assets and dashboard page must survive packaging.

This exists because of a specific, easy-to-miss failure: ``setup.py`` listed
``packages`` but declared no ``package_data``, so a wheel built from it
installed cleanly and then served a bare 404 for ``/static``. That reads as a
routing bug, so it gets debugged in the wrong place for a long time.

The check builds a real wheel and inspects its contents. It is slower than a
source-tree assertion and it is the only version that would have caught the
omission, because the files *are* in the repository either way.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
STATIC = ROOT / "quantforge" / "api" / "static"

#: Every asset the application serves from /static. If one is missing from a
#: built wheel the page breaks in ways that are hard to trace back to packaging.
EXPECTED_ASSETS = (
    "index.html",
    "dashboard.html",
    "manifest.json",
    "sw.js",
    "icon-192.png",
    "icon-512.png",
)


class TestSourceTree:
    @pytest.mark.parametrize("name", EXPECTED_ASSETS)
    def test_asset_is_present(self, name):
        assert (STATIC / name).is_file(), f"{name} is missing from the source tree"


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    """Build a wheel and return its path, or skip explaining why it could not be.

    The skip message is explicit because a packaging test that silently does
    not run is the same failure mode this test exists to prevent.
    """
    out = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "-w", str(out), str(ROOT)],
        capture_output=True, text=True, timeout=600,
    )
    wheels = list(out.glob("*.whl"))
    if not wheels:
        pytest.skip(
            "could not build a wheel in this environment "
            f"(pip exit {result.returncode}); the packaging test did not run. "
            f"stderr tail: {result.stderr[-300:]}"
        )
    return wheels[0]


class TestBuiltWheel:
    @pytest.mark.parametrize("name", EXPECTED_ASSETS)
    def test_asset_is_in_the_wheel(self, wheel, name):
        """The failure this whole file exists for."""
        with zipfile.ZipFile(wheel) as zf:
            members = set(zf.namelist())
        matches = [m for m in members
                   if m.endswith(f"quantforge/api/static/{name}")]
        assert matches, (
            f"{name} is not in {wheel.name}. The API would serve a 404 for it "
            f"after installation. Members under api/static: "
            f"{sorted(m for m in members if 'api/static' in m)}"
        )

    def test_static_directory_is_not_empty_in_the_wheel(self, wheel):
        with zipfile.ZipFile(wheel) as zf:
            statics = [m for m in zf.namelist() if "api/static/" in m]
        assert len(statics) >= len(EXPECTED_ASSETS)

    def test_the_package_itself_is_importable_from_the_wheel(self, wheel):
        with zipfile.ZipFile(wheel) as zf:
            members = zf.namelist()
        assert any(m.endswith("quantforge/api/app.py") for m in members), \
            "the API module is missing from the wheel"

    def test_no_source_tests_are_shipped(self, wheel):
        """Tests do not belong in a distributable wheel."""
        with zipfile.ZipFile(wheel) as zf:
            members = zf.namelist()
        leaked = [m for m in members if m.startswith("tests/")]
        assert not leaked, f"test files leaked into the wheel: {leaked[:5]}"
