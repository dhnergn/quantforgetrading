"""Main entry point for QuantForge AI signal bot.

Live trading is DISABLED. This is a signal generation and backtesting
platform for research purposes only.

DISCLAIMER: No strategy, indicator, or ML model guarantees profits or
predicts markets with certainty. A profitable historical backtest is
evidence about the tested historical assumptions, not proof of future
profitability.
"""
from __future__ import annotations

import sys


def main() -> None:
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # Run API server
        import uvicorn
        from quantforge.api.main import create_app

        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run CLI
        from quantforge.cli import main as cli_main
        cli_main()


if __name__ == "__main__":
    main()
