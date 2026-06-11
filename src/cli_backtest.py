"""CLI entry point for World Cup backtesting.

Usage:
    uv run python -m src.cli_backtest --year 2022
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="World Cup Backtester")
    parser.add_argument("--year", type=int, default=2022, help="World Cup year to backtest")
    args = parser.parse_args()

    settings_path = Path("config/settings.json")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))

    console = Console()

    from src.backtester import WorldCupBacktester
    bt = WorldCupBacktester(settings)
    results = bt.run_backtest(args.year)
    bt.print_report(results, console)


if __name__ == "__main__":
    main()
