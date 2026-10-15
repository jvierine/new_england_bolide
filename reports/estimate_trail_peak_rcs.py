"""Order-of-magnitude RCS of the strongest Figure 3 trail echoes."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reports.generate_beverly_rcs_range_path import load_station_gps
from reports.generate_station_head_rcs_figure import C_MPS, K_BOLTZMANN, geodetic_to_ecef


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bandwidth-hz", type=float, default=12.5)
    parser.add_argument("--out", type=Path, default=ROOT / "products/trail_peak_rcs.h5")
    args = parser.parse_args()
    fit_path = ROOT / "products/ballistic_head_trail_correlated_fit_v0p2p9_rinterp4_zp16_3incoh/ballistic_head_trail_correlated_fit.json"
    points_path = ROOT / "products/trail_echo_roi/trail_echo_roi_points.json"
    model = json.loads(fit_path.read_text())["model"]
    positions = geodetic_to_ecef(np.array(model["lat_deg"]), np.array(model["lon_deg"]), np.array(model["alt_m"]))
    # Locate a stationary trail cell by its measured bistatic range, not by
    # the later observation time, when the body has already passed that cell.
    keep = np.array(model["alt_m"]) >= 33610.
    positions = positions[keep]
    altitude = np.array(model["alt_m"])[keep]
    rows = json.loads(points_path.read_text())["points"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as out:
        out.attrs["fit_source"] = str(fit_path)
        out.attrs["snr_source"] = str(points_path)
        out.attrs["noise_bandwidth_hz"] = args.bandwidth_hz
        out.attrs["system_temperature_k"] = 1.54e4
        out.attrs["power_per_transmitter_w"] = 500.
        out.attrs["effective_tx_gain_linear"] = 1.
        out.attrs["effective_rx_gain_linear"] = 1.
        out.attrs["gain_convention"] = "Unity per-channel effective gains; no coherent array or polarization-combination gain for the power-averaged Figure 3 product"
        out.attrs["interpretation"] = "Approximate peak-bin RCS inferred from Doppler-whitened SNR; not an absolute calibration or total trail-integrated RCS"
        for station in sorted({r["station"] for r in rows}):
            peak = max((r for r in rows if r["station"] == station and r["unix_time"] >= 1780164380.), key=lambda r: r["snr_db"])
            gps_tx, gps_rx, frequency = load_station_gps(ROOT / "2026_05_30_bolide", station)
            tx, rx = [geodetic_to_ecef(*[np.array([v]) for v in gps])[0] for gps in [gps_tx, gps_rx]]
            rt, rr = [np.linalg.norm(positions - site, axis=1) for site in [tx, rx]]
            j = np.argmin(abs(rt + rr - peak["range_km"] * 1000.))
            sigma = 10. ** (peak["snr_db"] / 10.) * K_BOLTZMANN * 1.54e4 * args.bandwidth_hz * (4. * np.pi) ** 3 * rt[j] ** 2 * rr[j] ** 2 / (500. * (C_MPS / frequency) ** 2)
            g = out.create_group(station)
            for key, value in {"snr_db": peak["snr_db"], "rcs_m2": sigma, "rcs_dbsm": 10. * np.log10(sigma), "range_tx_m": rt[j], "range_rx_m": rr[j], "observed_bistatic_range_km": peak["range_km"], "model_altitude_m": altitude[j]}.items():
                g[key] = value
            print(f"{station}: SNR {peak['snr_db']:.2f} dB, Rt={rt[j]/1000:.1f} km, Rr={rr[j]/1000:.1f} km, RCS={10*np.log10(sigma):.2f} dBsm")


if __name__ == "__main__":
    main()
