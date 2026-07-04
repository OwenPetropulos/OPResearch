"""
run_all.py
Orchestrates the complete OPResearch data pipeline in dependency order.

Execution order:
  1. market_snapshot          — no dependencies
  2. macro_and_sector_news    — no dependencies (uses multi-source aggregation)
  3. sectors_overview         — depends on sector_news output
  4. portfolio_prices         — no dependencies
  5. earnings_calendar        — no dependencies
  6. ma_deals                 — uses news aggregation (optional, can fail)

Each script runs as a subprocess; failures are isolated but logged.
Exit code 1 if any critical script fails, 0 if all critical scripts succeed.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

SCRIPTS_DIR = Path(__file__).resolve().parent

# Critical scripts (pipeline fails if any of these fail)
CRITICAL_SCRIPTS = [
    "generate_market_snapshot.py",
    "generate_macro_and_sector_news.py",
    "generate_sectors_overview.py",
    "generate_portfolio_prices.py",
]

# Optional scripts (pipeline succeeds even if these fail)
OPTIONAL_SCRIPTS = [
    "generate_earnings_calendar.py",
    "generate_ma_deals.py",
]

def run_script(script_name: str, is_critical: bool = True) -> bool:
    """Run a single pipeline script. Returns True on success."""
    script_path = SCRIPTS_DIR / script_name
    
    print(f"\n{'='*70}")
    print(f"Running: {script_name} {'[CRITICAL]' if is_critical else '[OPTIONAL]'}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=False,
            timeout=180,  # 3 minutes per script
        )
        
        if result.returncode == 0:
            print(f"[OK] {script_name} completed successfully.")
            return True
        else:
            status = "[FAIL]" if is_critical else "[WARN]"
            print(f"{status} {script_name} exited with code {result.returncode}.")
            return False
    
    except subprocess.TimeoutExpired:
        status = "[TIMEOUT FAIL]" if is_critical else "[TIMEOUT WARN]"
        print(f"{status} {script_name} exceeded 180 second limit.")
        return False
    
    except Exception as e:
        status = "[ERROR FAIL]" if is_critical else "[ERROR WARN]"
        print(f"{status} {script_name} raised exception: {e}")
        return False

def main():
    print(f"\n{'='*70}")
    print(f"OPResearch Data Pipeline — Starting at {datetime.utcnow().isoformat()}")
    print(f"{'='*70}")
    
    results = {}
    
    # Run critical scripts
    for script in CRITICAL_SCRIPTS:
        ok = run_script(script, is_critical=True)
        results[script] = ok
    
    # Run optional scripts
    for script in OPTIONAL_SCRIPTS:
        ok = run_script(script, is_critical=False)
        results[script] = ok
    
    # Summary
    print(f"\n{'='*70}")
    print("Pipeline Summary")
    print(f"{'='*70}")
    
    critical_ok = all(results.get(s, False) for s in CRITICAL_SCRIPTS)
    
    for script, ok in results.items():
        is_critical = script in CRITICAL_SCRIPTS
        status = "[OK]" if ok else "[FAIL]"
        label = "[CRITICAL]" if is_critical else "[OPTIONAL]"
        print(f"  {status} {label} {script}")
    
    print()
    
    if critical_ok:
        print(f"All critical scripts completed successfully.")
        print(f"Pipeline finished at {datetime.utcnow().isoformat()}")
        sys.exit(0)
    else:
        failed = [s for s, ok in results.items() if not ok and s in CRITICAL_SCRIPTS]
        print(f"Critical scripts failed: {', '.join(failed)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
