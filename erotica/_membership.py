"""Una sola forma de leer una columna de pertenencia, porque hay **dos** y no significan lo mismo.

El paquete escribe tres columnas de probabilidad y **cuáles existen depende de qué camino corriste**:

===========================================  ==========================================
camino                                       columnas que quedan en ``clustering.data``
===========================================  ==========================================
:meth:`Clustering.search`                    ``probability_hdbscan`` **y sólo esa**
                                             (``core/clustering.py:1267``)
:meth:`Clustering.search_pseudoprobability`  las tres: ``probability_hdbscan``,
                                             ``probability_times`` y ``probability``
                                             (``core/clustering.py:773-775``)
===========================================  ==========================================

``probability`` es ``probabilities_ × probability_times``, así que es **siempre ≤**
``probability_hdbscan``: el mismo umbral sobre una u otra selecciona conjuntos distintos.

**El defecto que esto arregla no era el nombre: era el silencio.** Cuatro sitios pedían la columna
con ``if umbral is not None and columna in table.colnames``, o sea si la columna no estaba **se
saltaban el filtro y devolvían la tabla entera sin decir nada**. Medido ejercitando
:func:`erotica.selection.census_detectability_from_members` sobre
``data/test/NGC6383/comments_paper/cluster_data.ecsv`` —331 filas, salida de ``search()``, sin
columna ``probability``— con ``probability_threshold=0.6``:

===============================  =======================
``probability_column``           ``n_stars`` que usó
===============================  =======================
``"probability"`` (ausente)      **331** — la tabla entera
``"probability_hdbscan"``        **264**
===============================  =======================

67 estrellas, el 20%, entrando calladas en un censo que pidió miembros sobre 0,6.

**Lo que NO se unifica, a propósito.** El ajuste de isócrona lee ``probability_hdbscan`` y el resto
del paquete lee ``probability``. Esa asimetría se conserva porque **el manuscrito de NGC 6383 está
enviado y sus números no se mueven mientras esté en revisión**; cambiar la columna por defecto los
cambiaría. Lo que sí cambia es que deja de estar escondida en una línea: ahora es un parámetro con
nombre en la firma, y esta tabla dice cuál es cuál.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astropy.table import QTable

#: El que usa el ajuste de isócrona. Ver el docstring del módulo: no es un capricho, es P01.
COLUMNA_ISOCRONA = "probability_hdbscan"
#: El que usa todo lo demás: ``probabilities_ × probability_times``.
COLUMNA_POR_DEFECTO = "probability"


def select_by_probability(table: QTable, column: str, threshold: float | None) -> QTable:
    """``table`` filtrada a ``table[column] >= threshold``.

    ``threshold=None`` devuelve la tabla intacta — eso es pedir explícitamente no filtrar. Una
    columna ausente con un umbral pedido es un **error**, nunca un no-filtro: devolver el catálogo
    entero cuando alguien pidió miembros sobre 0,6 es el fallo silencioso que este módulo existe
    para matar.
    """
    if threshold is None:
        return table
    if column not in table.colnames:
        raise KeyError(
            f"La tabla no tiene la columna {column!r}, así que no se puede aplicar "
            f"el umbral {threshold}. `Clustering.search()` escribe sólo "
            f"'probability_hdbscan'; 'probability' y 'probability_times' las escribe "
            f"`Clustering.search_pseudoprobability()`. Corre ésa, o pasa "
            f"`probability_column='probability_hdbscan'`. Columnas presentes: "
            f"{[c for c in table.colnames if c.startswith('probability')]}"
        )
    return table[table[column] >= threshold]
