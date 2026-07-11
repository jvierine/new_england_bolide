# Marc Fries dark-flight products

These external-format inputs were supplied by Marc Fries for the Cape Cod Bay
weather-radar dark-flight analysis.

- `cape_cod_bay_dark_flight_20260626.kmz` is the latest supplied recovery-model
  KMZ (`MA Cape Cod Bay 30 May 2026 1806 UTC 04.kmz`).
- `cape_cod_bay_dark_flight_20260625.json` is the accompanying structured model
  output extracted from `fwexternalccb.zip`.

`reports/plot_fragment_size_hypotheses.py` reads the predicted flight paths
from the KMZ and the weather-radar centroids from the JSON. These formats are
retained because they are the documented external source products.
