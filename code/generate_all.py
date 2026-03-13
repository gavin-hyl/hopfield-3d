#!/usr/bin/env python3
"""Generate all figures for the final report."""

import os
import subprocess
import sys

SCRIPTS = [
    "fig_picard.py",
    "fig_sensitivity.py",
    "fig_equilibria.py",
    "fig_iss.py",
    "fig_barrier.py",
    "fig_contraction.py",
    "fig_hybrid.py",
    "fig_bifurcation.py",
]


def main():
    os.makedirs("../assets", exist_ok=True)
    for script in SCRIPTS:
        print(f"--- Running {script} ---")
        result = subprocess.run([sys.executable, script], capture_output=True, text=True)
        if result.stdout:
            print(result.stdout.strip())
        if result.returncode != 0:
            print(f"ERROR in {script}:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
    print("--- All figures generated ---")


if __name__ == "__main__":
    main()
