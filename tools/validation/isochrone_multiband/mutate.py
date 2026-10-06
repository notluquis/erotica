"""Mutation run of tests/test_isochrone_multiband.py, on a COPY of the package (here /tmp/mut_mb_47555,
made with cp -R of erotica/ and the test file) so the running experiments never import a mutant.
Each mutation is applied, checked to be in the file, the named test run, the original restored;
the last entry is the restored suite. Output: mutations.json (committed copy next to this file).
"""

import json
import shutil
import subprocess
import sys

M = "/tmp/mut_mb_47555"
F = f"{M}/erotica/analysis/_isochrone_multiband.py"
orig = open(F).read()
shutil.copy(F, F + ".orig")
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
            "HOME": "/Users/notluquis",
        },
    )
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
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
        "HOME": "/Users/notluquis",
    },
)
res.append(
    {"restored_suite_green": r.returncode == 0, "summary": r.stdout.strip().splitlines()[-1]}
)
print(res[-1])
json.dump(res, open(f"{M}/mutations.json", "w"), indent=1)
