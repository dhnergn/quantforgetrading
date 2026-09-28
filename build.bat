@echo off
REM Cross-platform build script for Windows

echo Building QuantForge AI...

REM Install dependencies
pip install -r requirements.txt
pip install -e .

REM Run tests
python -m pytest tests/ -q

echo Build complete!
