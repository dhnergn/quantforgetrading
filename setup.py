"""Packaging for QuantForge AI."""
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

#: The packages that make up the platform. Listed explicitly rather than via
#: find_packages() so that a stray directory cannot silently become part of the
#: distributed package.
#: The packages that make up the platform. Listed explicitly rather than via
#: find_packages() so that a stray directory cannot silently become part of the
#: distributed package. Every entry is asserted to exist by the packaging test,
#: so a stale entry fails the build rather than the install.
PACKAGES = [
    "quantforge",
    "quantforge.core",
    "quantforge.data",
    "quantforge.analysis",
    "quantforge.quant",
    "quantforge.features",
    "quantforge.strategy",
    "quantforge.backtest",
    "quantforge.portfolio",
    "quantforge.validation",
    "quantforge.ml",
    "quantforge.risk",
    "quantforge.execution",
    "quantforge.execution.brokers",
    "quantforge.monitoring",
    "quantforge.security",
    "quantforge.persistence",
    "quantforge.api",
]

setup(
    name="quantforge-ai",
    version="0.1.0",
    description=(
        "Research-grade quantitative trading platform. Live trading is disabled "
        "by default and requires explicit, signed authorisation."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    packages=PACKAGES,
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24,<3.0",
        "pandas>=2.0,<3.0",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "SQLAlchemy>=2.0",
        "fastapi>=0.104",
        "pydantic>=2.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        # argon2 is preferred for password hashing; the PBKDF2 fallback is
        # portable rather than a silent downgrade, and the active algorithm is
        # always reported by PasswordHasher.algorithm.
        "security": ["argon2-cffi>=21.1"],
        "server": ["uvicorn>=0.24", "httpx>=0.25"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "ruff>=0.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
