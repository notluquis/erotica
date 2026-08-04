"""Run pyUPMASK (Pera et al. 2021) as an external script and return co-indexed probabilities.

WHY THIS EXISTS
---------------
An earlier audit in this programme recorded that ``pyupmask`` is "not pip-installable" and
the three-way benchmark was therefore scoped down to two methods. The first half of that is
**confirmed** and the second half is **not a consequence of it**:

* ``pip index versions`` returns *No matching distribution found* for ``pyupmask``,
  ``pyUPMASK``, ``py-upmask`` and ``upmask``; ``pypi.org/pypi/pyupmask/json`` and
  ``pypi.org/pypi/pyUPMASK/json`` both return **HTTP 404**. There is no PyPI release.
* The upstream repository ``Gabriel-p/pyUPMASK`` is **archived**, last pushed
  2021-06-03, and contains no ``setup.py`` / ``pyproject.toml`` / ``requirements.txt`` —
  it ships as ``pyUPMASK.py`` plus a ``modules/`` package and a ``params.ini``.

But *not installable* is not the same as *not runnable*. Cloned and driven as a script on
python 3.13 the modules import and the pipeline executes: its only removed-stdlib
dependency, ``from distutils.util import strtobool`` in ``modules/dataIO.py``, resolves
through setuptools' bundled ``_distutils`` shim. So the third baseline can be measured
after all, and this module does it.

CONTRACT
--------
``run_pyupmask(realisation, repo, ...) -> (probabilities, seconds, notes)``

* pyUPMASK reads every file in ``<repo>/input`` and writes ``<repo>/output/<name>``.
  Each call gets a private working copy of the repo tree so concurrent cells cannot
  collide over those fixed directories.
* Row order is preserved end to end: ``dread`` reads the ascii table in file order,
  ``dwrite`` writes ``full_data`` in that same order with a ``probs_final`` column, and
  with no masked columns present ``data_rjct`` is empty so nothing is appended out of
  order. An explicit ``ID`` column is written and checked on read-back rather than
  trusted — co-indexing is the assumption the whole harness rests on.
* Stars rejected by pyUPMASK's own 5-sigma outlier stage carry ``probs_final = -1``.
  Those are **remapped to 0.0** (an explicit non-member) and counted in ``notes``;
  leaving -1 in place would corrupt Brier and ECE.

WHAT WOULD FALSIFY A RESULT FROM HERE
-------------------------------------
If the ``ID`` round-trip check fails, or if the number of returned rows differs from the
number written, the probabilities are not co-indexed and every score computed from them is
meaningless. Both are checked and raise rather than warn.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_REPO = Path(
    "/private/tmp/claude-501/-Users-notluquis-phd/"
    "9a0f9b71-83f6-46ec-ba5e-2fc967bc6851/scratchpad/pyUPMASK"
)


def _patch_ini(
    text: str, *, ol_runs: int, il_runs: int, n_membs: int, seed: int, clust_method: str
) -> str:
    """Set the knobs this benchmark controls; leave every other default untouched."""
    subs = {
        r"^rnd_seed\s*=.*$": f"rnd_seed     =  {seed}",
        r"^verbose\s*=.*$": "verbose      = 0",
        r"^ID\s*=.*$": "ID           = ID",
        r"^xy_coords\s*=.*$": "xy_coords    = _x  _y",
        r"^data\s*=.*$": "data         = pmRA  pmDE  Plx",
        r"^OL_runs\s*=.*$": f"OL_runs      = {ol_runs}",
        r"^IL_runs\s*=.*$": f"IL_runs    = {il_runs}",
        r"^N_membs\s*=.*$": f"N_membs    = {n_membs}",
        r"^clust_method\s*=.*$": f"clust_method = {clust_method}",
    }
    for pat, rep in subs.items():
        text = re.sub(pat, rep, text, count=1, flags=re.M)
    return text


def run_pyupmask(
    real,
    *,
    repo: Path = DEFAULT_REPO,
    workdir: Path,
    ol_runs: int = 25,
    il_runs: int = 25,
    n_membs: int = 25,
    seed: int = 0,
    clust_method: str = "GaussianMixture",
    timeout_s: int = 1800,
) -> tuple[np.ndarray, float, dict]:
    """Run pyUPMASK on one :class:`Realisation`; return (probabilities, seconds, notes)."""
    from astropy.io import ascii as ap_ascii
    from astropy.table import Table

    n = real.truth.size
    work = Path(workdir)
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(repo, work, ignore=shutil.ignore_patterns(".git", "output"))
    (work / "input").mkdir(exist_ok=True)
    for f in (work / "input").glob("*"):
        if f.is_file():
            f.unlink()
    (work / "output").mkdir(exist_ok=True)

    ids = np.arange(1, n + 1)
    tbl = Table(
        {
            "ID": ids,
            "_x": real.ra,
            "_y": real.dec,
            "pmRA": real.pmra,
            "pmDE": real.pmdec,
            "Plx": real.plx,
            "e_pmRA": real.e_pmra,
            "e_pmDE": real.e_pmdec,
            "e_Plx": real.e_plx,
        }
    )
    in_path = work / "input" / "cell.dat"
    ap_ascii.write(tbl, in_path, format="basic", overwrite=True)

    ini = work / "params.ini"
    ini.write_text(
        _patch_ini(
            ini.read_text(),
            ol_runs=ol_runs,
            il_runs=il_runs,
            n_membs=n_membs,
            seed=seed,
            clust_method=clust_method,
        )
    )

    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "pyUPMASK.py"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    secs = time.perf_counter() - t0
    out_path = work / "output" / "cell.dat"
    if proc.returncode != 0 or not out_path.exists():
        return (
            np.zeros(n),
            secs,
            {
                "error": f"returncode={proc.returncode}",
                "stderr": proc.stderr[-2000:],
                "stdout": proc.stdout[-1000:],
            },
        )

    res = Table.read(out_path, format="ascii")
    if len(res) != n:
        raise RuntimeError(
            f"pyUPMASK returned {len(res)} rows for {n} inputs; co-indexing is broken."
        )
    if not np.array_equal(np.asarray(res["ID"], dtype=int), ids):
        raise RuntimeError(
            "pyUPMASK row order does not match the input ID order; co-indexing is broken."
        )
    probs = np.asarray(res["probs_final"], dtype=float)
    n_rejected = int((probs < 0).sum())
    probs = np.where(probs < 0, 0.0, probs)
    return (
        probs,
        secs,
        {
            "n_outlier_rejected": n_rejected,
            "ol_runs": ol_runs,
            "il_runs": il_runs,
            "clust_method": clust_method,
        },
    )
