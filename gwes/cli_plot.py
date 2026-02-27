#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from importlib import resources


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="gwes-plot",
        description="Run the bundled GWES plotting R script via Rscript."
    )
    # pass through everything after '--' or just pass unknown args to R
    ap.add_argument("--rscript", default="Rscript", help="Path to Rscript binary (default: Rscript)")
    ap.add_argument("--print-script-path", action="store_true", help="Print the path to the bundled R script and exit")
    args, rest = ap.parse_known_args()

    rbin = args.rscript
    if shutil.which(rbin) is None and rbin == "Rscript":
        print("[error] Rscript not found in PATH. Install R or pass --rscript /path/to/Rscript.", file=sys.stderr)
        raise SystemExit(2)

    # Locate packaged R script
    r_path = resources.files("scripts").joinpath("gwes_plotting.r")
    if args.print_script_path:
        print(str(r_path))
        return

    cmd = [rbin, str(r_path)] + rest
    try:
        p = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print(f"[error] Could not execute: {rbin}", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(p.returncode)
