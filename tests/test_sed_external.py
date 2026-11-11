"""The SED generator of ``tools/validation/sed_external`` (``sedgrid.truth_mags``) against closed
forms that do not go through its interpolation code.

1. ``eem`` at A_V = 0 on an EEM table row: the colours are the table's own identities
   (BP - RP = Bp-Rp, G - RP = G-Rp, G - Ks = V-Ks + G-V, J - H, H - Ks), read here from the text.
2. Extinction is additive and the same in both models: ``eem(A_V) - eem(0) = mist(A_V) - mist(0)``.

Needs the pinned EEM table and MIST BC tarball in ``sed_external/dl/`` (gitignored, fetched by the
scripts); skips without them, so CI does not download 0.5 GB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SED = Path(__file__).resolve().parents[1] / "tools/validation/sed_external"
pytestmark = pytest.mark.skipif(
    not ((SED / "dl/EEM.txt").exists() and (SED / "dl/UBVRIplus.txz").exists()),
    reason="EEM table / MIST BC tarball not in sed_external/dl (fetched by the scripts)",
)
sys.path.insert(0, str(SED))


def _eem_rows():
    """(Teff, {colour: value}) for complete rows, parsed here, not by sedgrid.load_eem."""
    head = None
    out = []
    for ln in (SED / "dl/EEM.txt").read_text().splitlines():
        if ln.startswith("#SpT"):
            head = ln.lstrip("#").split()
            continue
        if head is None or ln.startswith("#") or not ln.strip():
            if head is not None and ln.startswith("# UPDATES"):
                break
            continue
        r = dict(zip(head, ln.split(), strict=False))
        try:
            v = {k: float(r[k]) for k in ("Teff", "G-V", "Bp-Rp", "G-Rp", "V-Ks", "J-H", "H-Ks")}
        except (KeyError, ValueError):
            continue
        out.append(v)
    return out


def test_eem_colours_at_zero_extinction_are_the_table():
    import sedgrid as sg

    rows = [r for r in _eem_rows() if 3500 <= r["Teff"] <= 9000][::5]
    assert len(rows) >= 5
    for r in rows:
        m = np.ravel(sg.truth_mags("eem", np.log10(r["Teff"]), 4.5, 0.0))
        G, BP, RP, J, H, Ks = m
        np.testing.assert_allclose(BP - RP, r["Bp-Rp"], atol=1e-9)
        np.testing.assert_allclose(G - RP, r["G-Rp"], atol=1e-9)
        np.testing.assert_allclose(G - Ks, r["V-Ks"] + r["G-V"], atol=1e-9)
        np.testing.assert_allclose(J - H, r["J-H"], atol=1e-9)
        np.testing.assert_allclose(H - Ks, r["H-Ks"], atol=1e-9)


def test_extinction_is_the_same_additive_term_in_both_models():
    import sedgrid as sg

    for lt, lg, av in [(3.6, 4.5, 0.7), (3.75, 4.0, 1.3), (3.9, 4.5, 2.2)]:
        de = sg.truth_mags("eem", lt, lg, av) - sg.truth_mags("eem", lt, lg, 0.0)
        dm = sg.truth_mags("mist", lt, lg, av) - sg.truth_mags("mist", lt, lg, 0.0)
        np.testing.assert_allclose(de, dm, atol=1e-12)
        assert np.all(np.ravel(de) > 0)  # dimmer in every band
