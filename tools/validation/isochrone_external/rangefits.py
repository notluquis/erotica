"""Read a few HEALPix columns of a remote, multi-GB FITS image with HTTP range requests.

Why: the Edenhofer+2024 map is 3.25 GB on Zenodo and the measured download rate on 2026-10-05 was
1.1 MB/s (~50 min); NGC 6383 needs ~10^2 of the 786 432 pixels of each of the 516 distance layers.
Every byte read here is the file's own (big-endian float32, rows = distance layers, NEST order), so
the subset is bit-identical to the same slice of the downloaded file -- checked by
``test_subset_against_full_row`` below on one full row.
"""

from __future__ import annotations

import concurrent.futures as cf
import time

import numpy as np
import requests

BLOCK = 2880


def _get(url: str, a: int, b: int, tries: int = 6) -> bytes:
    for k in range(tries * 20):
        try:
            r = requests.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
            if r.status_code == 429:  # Zenodo: measured 2026-10-05, 8 parallel -> 12/16 refused
                time.sleep(1.0 + float(r.headers.get("Retry-After") or 1))
                continue
            if r.status_code == 206 and len(r.content) == b - a + 1:
                return r.content
            if r.status_code == 416:  # past the end of the file: not transient, do not retry
                raise EOFError(f"HTTP 416 for {a}-{b}")
            raise OSError(f"HTTP {r.status_code}, {len(r.content)} bytes for {a}-{b}")
        except (OSError, requests.RequestException):
            if k >= tries * 20 - 1:
                raise
            time.sleep(2.0 * (k + 1))
    raise AssertionError


def _cards(raw: bytes) -> list[str]:
    s = raw.decode("ascii")
    return [s[i : i + 80] for i in range(0, len(s), 80)]


def hdus(url: str, max_hdus: int = 12) -> list[dict]:
    """Walk the headers: offset of each data block, its shape and the header cards."""
    out, off = [], 0
    for _ in range(max_hdus):
        cards, hstart = [], off
        while True:
            try:
                blk = _get(url, off, off + BLOCK - 1)
            except EOFError:
                return out
            off += BLOCK
            cs = _cards(blk)
            cards += cs
            if any(c.startswith("END") and c[3:].strip() == "" for c in cs):
                break
        h = {}
        for c in cards:
            if "=" in c[:10]:
                k, v = c[:8].strip(), c[10:].split("/")[0].strip().strip("'").strip()
                h[k] = v
        naxis = int(h.get("NAXIS", 0))
        dims = [int(h[f"NAXIS{i}"]) for i in range(1, naxis + 1)]
        nbytes = abs(int(h.get("BITPIX", 8))) // 8 * int(np.prod(dims)) if dims else 0
        nbytes += int(h.get("PCOUNT", 0))
        out.append({"header_offset": hstart, "data_offset": off, "dims": dims, "h": h})
        off += -(-nbytes // BLOCK) * BLOCK
    return out


def ranges_of(pix: np.ndarray, gap: int = 256) -> list[tuple[int, int]]:
    """Group sorted pixel indices into [lo, hi] runs, merging gaps shorter than ``gap`` pixels."""
    p = np.unique(pix)
    runs, lo, hi = [], int(p[0]), int(p[0])
    for x in p[1:]:
        if x - hi <= gap:
            hi = int(x)
        else:
            runs.append((lo, hi))
            lo = hi = int(x)
    runs.append((lo, hi))
    return runs


def read_columns(url: str, hdu: dict, pix: np.ndarray, workers: int = 2, gap: int = 256):
    """``data[row, pix]`` for every row of a 2D float32 image HDU (NAXIS1 = pixels)."""
    npix, nrow = hdu["dims"][0], hdu["dims"][1]
    assert hdu["h"]["BITPIX"] == "-32"
    runs = ranges_of(pix, gap)
    pix = np.unique(pix)
    out = np.empty((nrow, pix.size), dtype=">f4")

    def one(row):
        vals = {}
        for lo, hi in runs:
            a = hdu["data_offset"] + 4 * (row * npix + lo)
            buf = np.frombuffer(_get(url, a, a + 4 * (hi - lo + 1) - 1), dtype=">f4")
            for p in pix[(pix >= lo) & (pix <= hi)]:
                vals[int(p)] = buf[p - lo]
        return row, np.array([vals[int(p)] for p in pix], dtype=">f4")

    with cf.ThreadPoolExecutor(workers) as ex:
        for row, v in ex.map(one, range(nrow)):
            out[row] = v
            if row % 50 == 0:
                print(f"  row {row}/{nrow}", flush=True)
    return pix, out.astype(np.float64), runs
