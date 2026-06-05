import argparse
import pathlib
import json
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("mpc_file", default="mpcorb_extended.json")
args = parser.parse_args()
filename = pathlib.Path(args.mpc_file)
cache_file = filename.parent / f"{filename.name}.npy"

# Parameters to save
params = ["a", "e", "i", "Peri", "Node"]

if not cache_file.exists():
    with open(filename, "rb") as fh:
        mpc = json.load(fh)
    bodies = np.empty((6, len(mpc)), dtype=np.float64)
    for ind in range(len(mpc)):
        for k_ind, key in enumerate(params):
            bodies[k_ind, ind] = mpc[ind][key]
    np.save(cache_file, bodies)
else:
    bodies = np.load(cache_file)

bodies[0, :] = 1.0 / bodies[0, :]
params[0] = "1/a"

a, a_sig = 3.45, 0.88
a_inv = 1.0 / a
a_inv_sig = 0.5 * (1 / (a - a_sig) - 1 / (a + a_sig))

e, e_sig = 0.806, 0.041
inc, inc_sig = 19.5, 0.7
omega, omega_sig = 99.4, 2.2
Omega, Omega_sig = 68.9919, 0.0006

bolide = np.array([a_inv, e, inc, omega, Omega])

# All unique pairings
pairs = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
]

nrows, ncols = 3, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 16), layout="tight")
axes = axes.flatten()

for ax, (xpar, ypar) in zip(axes, pairs):
    ax.scatter(
        bodies[xpar, :],
        bodies[ypar, :],
        s=1,
        alpha=0.3,
        rasterized=True,
    )
    ax.plot(bolide[xpar], bolide[ypar], ".r")

    ax.set_xlabel(params[xpar])
    ax.set_ylabel(params[ypar])

fig.suptitle("MPCORB Orbital Element Pairwise Scatter Plots", fontsize=16)
plt.show()
