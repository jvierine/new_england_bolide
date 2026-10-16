"""Export the article's orbital elements and unchanged projected orbit curves."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pyorb
from astropy.time import Time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import plot_orbit_xy as source


def curve(kepler, count):
    return np.column_stack(source.kepler_to_xy(kepler, n_points=count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "reports/rust_orbits/input.h5")
    parser.add_argument("--from-archive", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "products/orbit_figure/scene.h5")
    parser.add_argument("--article", type=Path, default=ROOT.parent / "cape_cod_bay")
    args = parser.parse_args()
    if not args.from_archive:
        orbit_path = ROOT / source.DEFAULT_ORBIT
        macros_path = args.article / "reports/orbit_analysis_macros.tex"
        meteorites_path = args.article / "reports/meteorite_orbits_info_orbits.csv"
        samples, nominal, _, epoch_unix = source.read_kepler(orbit_path)
        metadata = source.read_orbit_metadata(orbit_path)
        macros = source.read_macro_values(macros_path)
        source.validate_macro_metadata(macros, metadata)
        assert samples.shape == (6, int(macros["OrbitSampleCount"]))
        assert np.isfinite(samples).all() and np.all(samples[0] > 0) and np.all(samples[1] < 1)
        meteorites = np.asarray(source.read_meteorite_fall_orbits(meteorites_path))
        assert meteorites.shape == (57, 6)
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.archive, "w") as h:
            h["samples_kepler"] = samples
            h["nominal_kepler"] = nominal
            h["meteorites_kepler"] = meteorites
            h.attrs["epoch_unix"] = epoch_unix
            h.attrs["trajectory_fit_version"] = metadata["trajectory_fit_version"]
            h.attrs["source_macro_text"] = macros_path.read_text()
            for name, path in [("orbit", orbit_path), ("meteorites", meteorites_path), ("macros", macros_path)]:
                h.attrs[f"{name}_source"] = str(path)
                h.attrs[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            for name in source.PLANETS:
                h[f"planets/{name}"] = source.planet_kepler(name, Time(epoch_unix, scale="utc", format="unix"))
        with h5py.File(args.archive) as h:
            np.testing.assert_array_equal(h["samples_kepler"][:], samples)
            np.testing.assert_array_equal(h["nominal_kepler"][:], nominal)
            np.testing.assert_array_equal(h["meteorites_kepler"][:], meteorites)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.archive) as inp, h5py.File(args.out, "w") as out:
        for key, value in inp.attrs.items():
            out.attrs[key] = value
        samples = inp["samples_kepler"][:]
        meteorites = inp["meteorites_kepler"][:]
        out["nominal_xy"] = curve(inp["nominal_kepler"][:], 1000)
        aphelion = inp["nominal_kepler"][:]
        aphelion[5] = 180.
        aphelion_orbit = pyorb.Orbit(M0=pyorb.M_sol, degrees=True, type="true", num=1)
        aphelion_orbit.kepler = aphelion[:, None]
        out["nominal_aphelion_xy"] = aphelion_orbit.cartesian[:2, 0] / source.AU
        for name, elements in [("samples_xy", samples.T), ("meteorites_xy", meteorites)]:
            xy = np.asarray([curve(k, 500) for k in elements])
            out.create_dataset(name, data=xy, compression="gzip", shuffle=True)
        for name, dataset in inp["planets"].items():
            kep = dataset[:]
            g = out.create_group(f"planets/{name}")
            g["xy"] = curve(kep, 900)
            orbit = pyorb.Orbit(M0=pyorb.M_sol, degrees=True, type="true", num=1)
            orbit.kepler = kep[:, None]
            g["position_xy"] = orbit.cartesian[:2, 0] / source.AU
        out.attrs["counts"] = json.dumps({"samples": samples.shape[1], "meteorites": len(meteorites)})
    with h5py.File(args.out) as h:
        with h5py.File(args.archive) as inp:
            np.testing.assert_array_equal(h["nominal_xy"][:], curve(inp["nominal_kepler"][:], 1000))
        for name, elements in [("samples_xy", samples.T), ("meteorites_xy", meteorites)]:
            for i, kep in enumerate(elements):
                np.testing.assert_array_equal(h[name][i], curve(kep, 500))
    print(f"Verified {samples.shape[1]} sample orbits and {len(meteorites)} meteorite orbits: {args.out}")


if __name__ == "__main__":
    main()
