"""Prepare the NOAA ABI/GLM observations and render them with Rustplotlib.

Run after downloading the ABI C02 scan and all three GLM granules for
18:06:00--18:07:00 UTC. See prepare_goes_glm_flash_density.py for selections.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    subprocess.run([sys.executable, str(ROOT / "reports/prepare_goes_glm_flash_density.py")], check=True)
    subprocess.run([
        "cargo", "run", "--release", "--manifest-path",
        str(ROOT / "reports/rust_goes/Cargo.toml"), "--",
        str(ROOT / "products/goes19_glm_20260530"),
    ], check=True)


if __name__ == "__main__":
    main()
