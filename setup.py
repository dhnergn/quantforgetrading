"""Setup script for QuantForge AI."""
from setuptools import find_packages, setup

setup(
    name="quantforge",
    version="0.1.0",
    description="Research-grade quantitative trading signal platform",
    author="QuantForge AI",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "sqlalchemy>=2.0.0",
        "pydantic>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "httpx>=0.25.0",
        "python-dateutil>=2.8.0",
        "PyYAML>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
)
