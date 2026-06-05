"""
Estimate Knudsen number
=============================================

Docstring for this example
"""

import numpy as np
import matplotlib.pyplot as plt
import metablate

alts = np.linspace(100e3, 30e3, num=200)

model = metablate.atmosphere.AtmPymsis()
data = model.density(
    time=np.datetime64("2018-07-28"),
    lat=np.array([69.0]),
    lon=np.array([12.0]),
    alt=alts,
)
atm_total_mass_density = data["Total"].values.flatten()
atm_total_number_density = np.zeros_like(atm_total_mass_density)
for symbol in model.species:
    s_dens = data[symbol].values.flatten()
    inds = np.logical_not(np.isnan(s_dens))
    atm_total_number_density[inds] = atm_total_number_density[inds] + s_dens[inds]

material_data = metablate.material.asteroidal
# Atmospheric mean free path, source: Physics Handbook p. 186
mfp_inf = 1.0 / (np.pi * (3.62e-10) ** 2 * atm_total_number_density)  # [m]

Kn_inf = 0.01

L = mfp_inf / Kn_inf


fig, ax = plt.subplots(figsize=(7, 4), layout="tight")

ax.semilogx(L, alts * 1e-3)
ax.set_xlabel("Size [m]")
ax.set_ylabel("Altitude [km]")

plt.show()

