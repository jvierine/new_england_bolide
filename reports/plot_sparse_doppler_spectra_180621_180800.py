from pathlib import Path
from datetime import datetime, timezone

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


PRODUCT_DIR = Path("products/trail_echo_sparse_deconv_full_3km_50code_rdmf_avx_dcfit_20260607")
OUT = Path("reports/figures/sparse_solver_memo/sparse_doppler_spectra_180621_180800.png")
STATIONS = ["beverly", "clinton", "pomfret", "amherstnh", "siena"]
START = datetime(2026, 5, 30, 18, 6, 21, tzinfo=timezone.utc).timestamp()
STOP = datetime(2026, 5, 30, 18, 8, 0, tzinfo=timezone.utc).timestamp()
BOLIDE_RANGE_KM = {
    "beverly": (190.0, 270.0),
    "clinton": (230.0, 285.0),
    "pomfret": (365.0, 415.0),
    "amherstnh": (255.0, 315.0),
    "siena": (385.0, 445.0),
}


def short_station_name(station: str) -> str:
    return {
        "amherstnh": "Amherst",
        "beverly": "Beverly",
        "clinton": "Clinton",
        "pomfret": "Pomfret",
        "siena": "Siena",
    }.get(station, station.title())


def load_station(station: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = PRODUCT_DIR / f"trail_sparse_deconv_{station}.h5"
    with h5py.File(path, "r") as h:
        time_unix = h["time_unix"][:]
        keep_time = np.where((time_unix >= START) & (time_unix < STOP))[0]
        doppler_hz = h["doppler_hz"][:]
        range_idx_grid = h["range_idx"][:]
        range_km = h["range_km"][:]
        support_idx = h["support_range_idx"][keep_time, :]
        support_voltage = h["support_voltage"][keep_time, :, :, :, :]
        support_count = h["support_count"][keep_time]

    spectra = np.zeros((len(keep_time), len(doppler_hz)), dtype=np.float64)
    range_power = np.zeros((len(keep_time), len(range_idx_grid)), dtype=np.float64)
    idx_to_dense = {int(v): i for i, v in enumerate(range_idx_grid)}
    range_min_km, range_max_km = BOLIDE_RANGE_KM[station]

    for row in range(len(keep_time)):
        for support_i in range(int(support_count[row])):
            range_idx = int(support_idx[row, support_i])
            if range_idx < 0:
                continue
            dense_i = idx_to_dense.get(range_idx)
            if dense_i is None:
                continue
            if not (range_min_km <= range_km[dense_i] <= range_max_km):
                continue
            coeff = support_voltage[row, :, :, :, support_i]
            if not np.any(coeff):
                continue
            spec = np.fft.fftshift(np.fft.fft(coeff, axis=0), axes=0)
            power = np.mean(np.abs(spec) ** 2, axis=(1, 2))
            spectra[row] += power
            range_power[row, dense_i] += float(np.sum(power))

    spectra_db = 10.0 * np.log10(np.maximum(spectra, 1e-30))
    return time_unix[keep_time], doppler_hz, spectra_db, range_km, range_power


def main() -> None:
    data = []
    all_db = []
    for station in STATIONS:
        item = load_station(station)
        data.append((station, *item))
        vals = item[2][np.isfinite(item[2])]
        if vals.size:
            all_db.append(vals)
    all_db = np.concatenate(all_db)
    vmax = float(np.nanpercentile(all_db, 99.0))
    vmin = vmax - 35.0

    fig, axes = plt.subplots(
        len(data),
        2,
        figsize=(12.5, 9.5),
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    if len(data) == 1:
        axes = axes[None, :]

    mesh = None
    for axrow, (station, times, doppler_hz, spectra_db, range_km, range_power) in zip(axes, data, strict=True):
        time_edges = np.r_[times - 0.25, times[-1] + 0.25]
        doppler_step = float(np.median(np.diff(doppler_hz)))
        doppler_edges = np.r_[doppler_hz - 0.5 * doppler_step, doppler_hz[-1] + 0.5 * doppler_step]
        time_edges_dt = [datetime.fromtimestamp(x, tz=timezone.utc) for x in time_edges]
        time_locator = mdates.SecondLocator(bysecond=range(0, 60, 20), tz=timezone.utc)
        time_formatter = mdates.DateFormatter("%H:%M:%S", tz=timezone.utc)

        ax = axrow[0]
        mesh = ax.pcolormesh(
            time_edges_dt,
            doppler_edges,
            spectra_db.T,
            shading="flat",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.axhline(0, color="w", lw=0.7, alpha=0.65)
        ax.set_ylabel(f"{short_station_name(station)}\nDoppler (Hz)")
        ax.set_ylim(-100, 100)
        ax.grid(True, color="w", alpha=0.18, linewidth=0.35)
        ax.xaxis.set_major_locator(time_locator)
        ax.xaxis.set_major_formatter(time_formatter)

        axr = axrow[1]
        range_power_db = np.full_like(range_power, np.nan, dtype=np.float64)
        keep = range_power > 0
        range_power_db[keep] = 10.0 * np.log10(range_power[keep])
        vals = range_power_db[np.isfinite(range_power_db)]
        if vals.size:
            rvmax = float(np.nanpercentile(vals, 99.0))
            rvmin = rvmax - 35.0
        else:
            rvmin, rvmax = 0.0, 1.0
        axr.pcolormesh(
            time_edges_dt,
            np.r_[range_km - 1.5, range_km[-1] + 1.5],
            range_power_db.T,
            shading="flat",
            cmap="viridis",
            vmin=rvmin,
            vmax=rvmax,
        )
        axr.set_ylabel("Range (km)")
        range_min_km, range_max_km = BOLIDE_RANGE_KM[station]
        axr.set_ylim(range_min_km, range_max_km)
        axr.grid(True, color="w", alpha=0.18, linewidth=0.35)
        axr.xaxis.set_major_locator(time_locator)
        axr.xaxis.set_major_formatter(time_formatter)

    axes[0, 0].set_title("Sparse Doppler spectra in bolide-related range band")
    axes[0, 1].set_title("Bolide-related active sparse range support")
    axes[-1, 0].set_xlabel("UTC on 2026-05-30")
    axes[-1, 1].set_xlabel("UTC on 2026-05-30")
    fig.colorbar(mesh, ax=axes[:, 0], label="Sparse coefficient spectral power (dB)", shrink=0.8)
    fig.suptitle("Low-altitude non-specular trail sparse spectra, 18:06:21-18:08:00 UTC", y=1.02)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
