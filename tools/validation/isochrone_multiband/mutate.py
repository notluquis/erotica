"""Mutation run of tests/test_isochrone_multiband.py, on a COPY of the package (a fresh temporary
directory holding erotica/, tests/conftest.py and the test file, copied from this checkout) so the
running experiments never import a mutant.
Each mutation is applied, checked to be in the file, the named test run, the original restored;
the last entry is the restored suite. Output: mutations.json (committed copy next to this file).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
M = tempfile.mkdtemp(prefix="mut_mb_")
shutil.copytree(ROOT / "erotica", f"{M}/erotica", ignore=shutil.ignore_patterns("__pycache__"))
os.makedirs(f"{M}/tests")
for t in ("conftest.py", "test_isochrone_multiband.py"):
    shutil.copy(ROOT / "tests" / t, f"{M}/tests/{t}")
F = f"{M}/erotica/analysis/_isochrone_multiband.py"
orig = open(F).read()
muts = [
    (
        "floor twice on colour",
        "1.0 / xp.sqrt(self._e_obs_col**2 + s2g)",
        "1.0 / xp.sqrt(self._e_obs_col**2 + 2 * s2g)",
        "reproduces_the_2d and single and one-sided",
    ),
    (
        "background over one coordinate",
        "np.log(span_g) + np.log(span_c))",
        "np.log(span_g))",
        "reproduces_the_2d and single and one-sided",
    ),
    (
        "Woodbury rank-1 correction dropped",
        "a = a - uy[0] ** 2 / det",
        "a = a",
        "reproduces_the_2d and binaries and one-sided",
    ),
    (
        "capacitance off-diagonal sign",
        "i00, i11, i01 = c11 / det, c00 / det, -c01 / det",
        "i00, i11, i01 = c11 / det, c00 / det, c01 / det",
        "segment_integral",
    ),
    (
        "zero point applied to every star",
        "x = x - zp[zi] * self._nir_group[j]",
        "x = x - zp[zi] * self._nir_has[j]",
        "zero_point",
    ),
    (
        "fitz19 at a constant colour",
        'kG = fitz19_k("G", col0, Av, xp)',
        'kG = fitz19_k("G", 0.0 * col0 + 1.0, Av, xp)',
        "per_point",
    ),
    (
        "missing band counted in the normalisation",
        "n_i = n_i + xp.where(sw[c] > 0, 1.0, 0.0)",
        "n_i = n_i + 1.0",
        "missing_for_every",
    ),
]
res = []
for name, old, new, k in muts:
    assert orig.count(old) == 1, name
    open(F, "w").write(orig.replace(old, new))
    assert new in open(F).read()
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"{M}/tests/test_isochrone_multiband.py",
            "-k",
            k,
        ],
        cwd=M,
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": M,
            "OMP_NUM_THREADS": "1",
            "PATH": "/usr/bin:/bin",
            "HOME": os.environ["HOME"],
        },
    )
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
    # a skipped test (no pytensor, missing data) is not a survived mutant: refuse to record it
    if "skipped" in last and "failed" not in last:
        shutil.rmtree(M)
        sys.exit(f"{name}: the selected test was skipped ({last}); run with the bayes extra installed")
    res.append({"mutation": name, "test": k, "red": r.returncode != 0, "summary": last})
    print(res[-1], flush=True)
    open(F, "w").write(orig)
assert open(F).read() == orig
r = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        f"{M}/tests/test_isochrone_multiband.py",
        "-k",
        "reproduces_the_2d or segment_integral or zero_point or per_point or missing_for_every",
    ],
    cwd=M,
    capture_output=True,
    text=True,
    env={
        "PYTHONPATH": M,
        "OMP_NUM_THREADS": "1",
        "PATH": "/usr/bin:/bin",
        "HOME": os.environ["HOME"],
    },
)
res.append(
    {"restored_suite_green": r.returncode == 0, "summary": r.stdout.strip().splitlines()[-1]}
)
print(res[-1])
json.dump(res, open(HERE / "mutations.json", "w"), indent=1)
shutil.rmtree(M)
