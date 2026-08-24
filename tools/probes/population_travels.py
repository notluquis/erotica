"""R8-12: que `population` lo produzca el modelo y lo REENVÍE el consumidor.

El guardia era `pytest -k wrapper`, que tarda 63 s aislado y pasa de 120 bajo contención — el
detector lo reportó como colgado, que es correcto pero no dice nada del hallazgo. Esto sondea las
dos mitades directamente: que las dos ramas declaren su familia, y que el único consumidor en
paquete la copie al resultado en vez de descartar `metadata` entero, que era el defecto.
"""

import sys

sys.path.insert(0, "/Users/notluquis/erotica")

import numpy as np  # noqa: E402
from astropy import units as u  # noqa: E402
from astropy.table import QTable  # noqa: E402

from erotica.analysis.inference import (  # noqa: E402
    ClusterInferenceAnalyzer,
    SamplingConfig,
    distance_model,
)

rng = np.random.default_rng(0)
n = 40
d = rng.normal(1.0, 0.02, n)
e = np.full(n, 0.05)
t = QTable(
    {
        "r_med_geo": d * u.kpc,
        "r_lo_geo": (d - e) * u.kpc,
        "r_hi_geo": (d + e) * u.kpc,
    }
)
cfg = SamplingConfig(
    draws=20, tune=20, chains=1, random_seed=1, progressbar=False, extra_kwargs={"cores": 1}
)

con = distance_model(t, distance_lo_column="r_lo_geo", distance_hi_column="r_hi_geo", sampling=cfg)
sin = distance_model(t, sampling=cfg)
assert con.metadata["population"] == "normal-marginalised", con.metadata
assert sin.metadata["population"] == "gamma", sin.metadata

# La mitad que era el defecto: el consumidor copiaba cuatro claves y tiraba `metadata`.
import inspect  # noqa: E402

src = inspect.getsource(ClusterInferenceAnalyzer.distance_and_parallax_by_probability)
assert '"population"' in src, "el consumidor no declara la clave population"
assert 'results["population"].append' in src, "el consumidor no reenvia la familia ajustada"

print("population: gamma / normal-marginalised, y el consumidor la reenvia")
