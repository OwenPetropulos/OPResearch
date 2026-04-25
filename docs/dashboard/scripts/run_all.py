"""
run_all.py

Orchestrates the full OPResearch data pipeline in the correct dependency order.

Order:
  1. market_snapshot    — no dependencies
  2. macro_news         — no dependencies
  3. sector_news        — no dependencies
  4. sectors_overview   — depends on sector_news.json
  5. portfolio_prices   — no dependencies

Runs each script as a subprocess so failures are isolated.
Exits with code 1 if any script fails, but always attempts all scripts.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

PIPELINE = [
    "generate_market_snapshot.py",
    "generate_macro_news.py",
    "generate_sector_news.py",
    "generate_sectors_overview.py",   # must run after generate_sector_news
    "generate_portfolio_prices.py",
]


def run_script(script_name: str) -> bool:
    """Run a single pipeline script. Returns True on success."""
    script_path = SCRIPTS_DIR / script_name
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=False,  # Don't raise — we handle return codes manually
            timeout=120,  # 2-minute timeout per script
        )
        if result.returncode == 0:
            print(f"[OK] {script_name} completed successfully.")
            return True
        else:
            print(f"[FAIL] {script_name} exited with code {result.returncode}.")
            return False
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {script_name} exceeded 120 second limit.")
        return False
    except Exception as e:
        print(f"[ERROR] {script_name} raised an exception: {e}")
        return False


def main():
    print("OPResearch Data Pipeline — Starting")
    results = {}

    for script in PIPELINE:
        ok = run_script(script)
        results[script] = ok

    # Summary
    print(f"\n{'='*60}")
    print("Pipeline Summary")
    print(f"{'='*60}")
    all_ok = True
    for script, ok in results.items():
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {script}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All scripts completed successfully.")
        sys.exit(0)
    else:
        print("One or more scripts failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
