"""Download grids into a user cache, outside the repository, with checksums.

The grid files are not EROTICA's to redistribute (only SPOTS declares a licence; see ``AGENTS.md``),
so they are fetched on demand into ``$EROTICA_GRIDS_CACHE`` (default ``~/.cache/erotica-grids``) and
checked:

* **SPOTS** -- against the MD5 the Zenodo API publishes for each file (an external checksum);
* **MIST v2.5, BHAC15** -- no checksum is published, so the SHA-256 measured on 2026-10-04 is pinned
  here. A mismatch raises :class:`ChangedUpstream` instead of silently using different models: a grid
  that changed upstream is a new version, and a fit must say which one it used.

``PARSEC`` is fetched through ``ezpadova`` (MIT; the CMD web-form client), an optional dependency:
the CMD form changes between releases and that client tracks it. Each call writes a
``manifest.json`` (URL, size, hash, date) next to the files.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import urllib.request
from pathlib import Path

MIST25_URL = "https://mist.science/data/tarballs_v2.5/isos/{system}.txz"
MIST25_SHA256 = {
    "UBVRIplus": "abf4c89c9091ca14f6c4d595d2c533cf1f240346ebb45532c3763ff53aa69543",
}
BHAC15_URL = "http://perso.ens-lyon.fr/isabelle.baraffe/BHAC15dir/{name}"
BHAC15_SHA256 = {
    "BHAC15_iso.GAIA": "43ba70b5ae87d32fdc2cd8b1346ad705b24c97d17b223510aa8a7bc8d753ab76",
    "BHAC15_iso.2mass": "e86198b15e0f28437f7ee83bcebe93cbbf2b02ee2ab6097295eee3155a28bcf2",
}
SPOTS_RECORD = "https://zenodo.org/api/records/3593339"


class ChangedUpstream(RuntimeError):
    """The file's hash differs from the one this version of EROTICA was validated against."""


def cache_dir() -> Path:
    d = Path(os.environ.get("EROTICA_GRIDS_CACHE", Path.home() / ".cache" / "erotica-grids"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hash(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(url: str, dest: Path, timeout: float = 600.0) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:  # noqa: S310
        while chunk := r.read(1 << 22):
            f.write(chunk)
    tmp.replace(dest)


def _manifest(d: Path, entries: list[dict]) -> None:
    m = d / "manifest.json"
    old = json.loads(m.read_text()) if m.exists() else []
    keep = {e["file"]: e for e in old}
    keep.update({e["file"]: e for e in entries})
    m.write_text(json.dumps(list(keep.values()), indent=1) + "\n")


def _checked(path: Path, expected: str | None, url: str, algo: str = "sha256") -> dict:
    got = _hash(path, algo)
    if expected is not None and got != expected:
        raise ChangedUpstream(f"{path.name}: {algo} {got} != pinned {expected} ({url})")
    return {
        "file": path.name,
        "url": url,
        "bytes": path.stat().st_size,
        algo: got,
        "checked_against": "pinned" if algo == "sha256" else "zenodo md5",
        "fetched": _dt.date.today().isoformat(),
    }


def fetch_spots(fspots: tuple[float, ...] = (0.0, 0.17, 0.34, 0.51, 0.68, 0.85)) -> Path:
    """SPOTS ``fXXX.isoc`` files from Zenodo 3593339, MD5-checked against the record."""
    d = cache_dir() / "spots"
    d.mkdir(exist_ok=True)
    with urllib.request.urlopen(SPOTS_RECORD, timeout=60) as r:  # noqa: S310
        files = {f["key"]: f for f in json.load(r)["files"]}
    entries = []
    for f in fspots:
        name = f"f{int(round(f * 100)):03d}.isoc"
        meta = files[name]
        dest = d / name
        if not dest.exists():
            _get(meta["links"]["self"], dest)
        e = _checked(dest, meta["checksum"].removeprefix("md5:"), meta["links"]["self"], "md5")
        e["license"] = "CC BY 4.0"
        entries.append(e)
    _manifest(d, entries)
    return d


def fetch_bhac15(names: tuple[str, ...] = ("BHAC15_iso.GAIA", "BHAC15_iso.2mass")) -> Path:
    d = cache_dir() / "bhac15"
    d.mkdir(exist_ok=True)
    entries = []
    for n in names:
        dest, url = d / n, BHAC15_URL.format(name=n)
        if not dest.exists():
            _get(url, dest)
        entries.append(_checked(dest, BHAC15_SHA256.get(n), url))
    _manifest(d, entries)
    return d


def fetch_mist25(system: str = "UBVRIplus", afe: str = "p0", vvcrit: str = "0.4") -> Path:
    """The v2.5 tarball of one photometric system (1.58 GB for UBVRIplus), and the files of one
    [alpha/Fe] and rotation extracted from it (15 [Fe/H] values, ~84 MB each)."""
    import tarfile

    base = cache_dir() / "mist_v2.5"
    raw = base / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    url = MIST25_URL.format(system=system)
    tar = raw / f"{system}.txz"
    if not tar.exists():
        _get(url, tar, timeout=3600)
    entry = _checked(tar, MIST25_SHA256.get(system), url)
    out = base / f"{system}_afe{afe.strip('p') or '0'}_vvcrit{vvcrit}"
    tag = f"_afe_{afe}_vvcrit{vvcrit}_"
    if not out.exists() or not any(out.iterdir()):
        out.mkdir(exist_ok=True)
        with tarfile.open(tar, "r:xz") as t:
            members = [m for m in t.getmembers() if tag in m.name and m.isfile()]
            for m in members:
                m.name = Path(m.name).name
                t.extract(m, out, filter="data")
    _manifest(base, [entry])
    return out


def fetch_parsec(out_name: str, **query) -> Path:
    """A PARSEC table from the CMD web form, through ``ezpadova.get_isochrones(**query)``.

    Example: ``fetch_parsec("v1.2S_gaiaEDR3.dat", logage=(6.0, 7.0, 0.05), MH=(-0.6, 0.3, 0.1),
    photsys_file="YBC_tab_mag_odfnew/tab_mag_gaiaEDR3.dat")``. The query is stored in the manifest.
    """
    try:
        from ezpadova import parsec
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("fetch_parsec needs ezpadova (pip install ezpadova, MIT)") from exc
    d = cache_dir() / "parsec"
    d.mkdir(exist_ok=True)
    raw = parsec.get_isochrones(return_df=False, **query)
    dest = d / out_name
    dest.write_bytes(raw)
    e = _checked(dest, None, "http://stev.oapd.inaf.it/cgi-bin/cmd")
    e["query"] = {k: list(v) if isinstance(v, tuple) else v for k, v in query.items()}
    _manifest(d, [e])
    return dest
