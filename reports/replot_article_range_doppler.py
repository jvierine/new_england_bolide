"""Replot article Figure 5 from the saved fit, without refitting any data."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fit_ballistic_head_trail_correlated_appendix as fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, default=fit.DEFAULT_OUT_DIR / 'ballistic_head_trail_correlated_fit.json')
    parser.add_argument('--out', type=Path, default=fit.ROOT / 'reports/figures/appendix_correlated_fit/ballistic_head_trail_correlated_range_doppler_fit.png')
    cli = parser.parse_args()
    payload = json.loads(cli.fit.read_text())
    args = argparse.Namespace(**payload['settings'])
    for key in ('head_selection_csv', 'data_dir', 'trail_json', 'initial_ballistic_json', 'combined_fit_json'):
        setattr(args, key, Path(getattr(args, key)))
    # Read exactly the archived selection referenced by this fit.
    obs = fit.load_selected_head_observations(args.head_selection_csv)
    assert len(obs) == payload['n_head_observations']
    geom = fit.station_geometry(args.data_dir, sorted({o['station'] for o in obs}))
    obs = fit.cache_head_station_states_gcrs(obs, geom)
    t_ref = payload['t_ref_unix']
    trail = fit.load_trail_positions(args.trail_json, args.trail_min_snr_db, args.max_aod_offset_deg, args.min_trail_alt_km, args.max_trail_alt_km)
    trail = [r for r in trail if not fit.is_excluded_trail_row(r)]
    initial = np.r_[fit.load_initial_state(args, t_ref), fit.fixed_log10_cd_area_over_mass(args)]
    times = np.r_[[o['unix_time'] for o in obs], payload['model']['time_unix']]
    density = fit.build_density_profile(times, trail, initial, t_ref)
    fit.plot_correlated_range_doppler(
        cli.out, np.asarray(payload['fit_parameters']), np.asarray(payload['fit_covariance']),
        obs, t_ref, density, args,
        excluded_doppler_keys={("clinton", 323)},
    )
    print(f'Replotted saved fit {payload["trajectory_fit_version"]}: {cli.out}')


if __name__ == '__main__':
    main()
