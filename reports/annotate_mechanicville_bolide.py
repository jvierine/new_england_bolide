#!/usr/bin/env python3
"""Launch the manual two-mask GUI for the Mechanicville Wyze frame."""

from __future__ import annotations

import sys

from annotate_mount_arlington_bolide import main


DEFAULT_ARGUMENTS = [
    "--image", "data/security_camera/mechanicville_20260530_140622.png",
    "--output", "annotations/mechanicville_bolide.h5",
    "--figure", "reports/figures/mechanicville_bolide_annotation.png",
    "--assumed-utc", "2026-05-30T18:06:22Z",
    "--latitude-deg", "42.0",
    "--longitude-deg", "-73.0",
    "--location-name", "Mechanicville, New York, US",
    "--title", "Mechanicville Wyze security-camera frame",
    "--crop", "1850", "2180", "250", "570",
]


if __name__ == "__main__":
    # Arguments typed by the user are appended and therefore override earlier
    # argparse values when the same option is repeated.
    sys.argv[1:1] = DEFAULT_ARGUMENTS
    main()
