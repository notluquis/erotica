"""Una columna de pertenencia ausente es un error, nunca un no-filtro.

Estos tests existen porque **la suite entera pasó verde con el arreglo puesto**: los cuatro sitios
que antes se saltaban el filtro en silencio no los ejercitaba nadie, así que el cambio de
comportamiento era invisible para los 552 tests. Eso no es que faltara un test, es que la suite no
podía ver la rama — el modo de falla de la lección 29 del `CLAUDE.md` global.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable

from erotica._membership import (
    COLUMNA_ISOCRONA,
    COLUMNA_POR_DEFECTO,
    select_by_probability,
)


def _tabla_de_search() -> QTable:
    """Lo que deja `Clustering.search()`: `probability_hdbscan` y nada más."""
    return QTable(
        {
            "probability_hdbscan": np.array([0.9, 0.7, 0.5, 0.3]),
            "parallax_error": np.array([0.02, 0.03, 0.04, 0.05]) * u.mas,
        }
    )


def test_columna_ausente_con_umbral_es_un_error_y_no_un_no_filtro():
    t = _tabla_de_search()
    with pytest.raises(KeyError) as exc:
        select_by_probability(t, COLUMNA_POR_DEFECTO, 0.6)
    # El mensaje tiene que nombrar la CAUSA -- que camino escribe que columna -- o quien lo lea
    # sabe que falta algo y no que hacer.
    mensaje = str(exc.value)
    assert "search_pseudoprobability" in mensaje
    assert "probability_hdbscan" in mensaje


def test_umbral_none_no_filtra_aunque_la_columna_falte():
    """Pedir explícitamente no filtrar es legal; es el silencio lo que no lo era."""
    t = _tabla_de_search()
    assert len(select_by_probability(t, COLUMNA_POR_DEFECTO, None)) == len(t)


def test_filtra_de_verdad_cuando_la_columna_esta():
    t = _tabla_de_search()
    assert len(select_by_probability(t, COLUMNA_ISOCRONA, 0.6)) == 2


def test_las_dos_columnas_por_defecto_son_distintas_a_proposito():
    """Si alguien las unifica, este test lo obliga a decirlo en voz alta.

    No se unifican porque el manuscrito de NGC 6383 está enviado: el ajuste de isócrona corre sobre
    `probability_hdbscan` y cambiar esa columna movería números que están en revisión.
    """
    assert COLUMNA_ISOCRONA != COLUMNA_POR_DEFECTO


def test_el_censo_ya_no_devuelve_el_catalogo_entero_en_silencio():
    """La regresión concreta, sobre la función real.

    Medido antes del arreglo sobre `cluster_data.ecsv` (331 filas, salida de `search()`, sin
    columna `probability`): con `probability_threshold=0.6` la función usaba **331** estrellas en
    vez de 264. 67 estrellas -- el 20% -- entraban calladas.
    """
    from erotica.selection import census_detectability_from_members

    t = _tabla_de_search()
    with pytest.raises(KeyError):
        census_detectability_from_members(
            t,
            significance_threshold=5.0,
            data_density=1.0 * u.deg**-2,
            probability_threshold=0.6,
            selection_function=lambda *a, **k: np.array([0.5]),
        )


def test_la_isocrona_expone_su_columna_en_la_firma():
    """La asimetría deja de estar escondida en el cuerpo de `setup`."""
    import inspect

    from erotica.analysis._isochrone import IsochroneFitter

    for metodo in (IsochroneFitter.setup, IsochroneFitter.cmd_distances):
        p = inspect.signature(metodo).parameters
        assert "probability_column" in p, f"{metodo.__name__} no expone la columna"
        assert p["probability_column"].default == COLUMNA_ISOCRONA
