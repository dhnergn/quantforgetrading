#!/usr/bin/env bash
# Linux and macOS entry point.
#
# Thin wrapper on purpose: the logic lives in scripts/bootstrap.py, which is
# the single implementation. Duplicating it in a shell and a batch file means
# two copies that drift, and bash is not even present on Windows.
set -euo pipefail
cd "$(dirname "$0")"
exec "${QUANTFORGE_PYTHON:-python3}" scripts/bootstrap.py "$@"
