#!/usr/bin/env python3
"""Compute a simple MSIS density ratio for the Cape Cod Bay bolide."""

import numpy as np
from pymsis import msis


DATE = np.array([np.datetime64("2026-05-30T18:06:20")])
LAT_DEG = np.array([42.0])
LON_DEG = np.array([-70.5])
ALT_KM = np.array([70.0, 120.0])

# PyMSIS bundled space-weather tables do not extend to this 2026 event yet.
F107 = 150.0
F107A = 150.0
AP = np.full((1, 7), 4.0)


def main() -> None:
    out = msis.run(DATE, LON_DEG, LAT_DEG, ALT_KM, f107s=F107, f107as=F107A, aps=AP)
    rho = np.squeeze(np.asarray(out, dtype=float))[..., 0]
    rho_70, rho_120 = rho
    print(f"rho(70 km)  = {rho_70:.6e} kg/m^3")
    print(f"rho(120 km) = {rho_120:.6e} kg/m^3")
    print(f"rho(70 km) / rho(120 km) = {rho_70 / rho_120:.3g}")


if __name__ == "__main__":
    main()
