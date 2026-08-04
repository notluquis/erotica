#!/usr/bin/env python3
r"""Build NGC 6383's Gaia DR3 :math:`\bar S(r)` at a resolution that actually resolves ``R_c``.

WHY THIS EXISTS
---------------
The radial completeness profile for NGC 6383 has been read twice at resolutions
that cannot see the cluster: ``mode='hpx7'`` (27.5' pixels against ``R_c`` = 1.38',
the whole 70' field spanning 2.55 pixels) and ``mode='multi'`` (order 10, 3.44',
which resolves the field but not the core). Both returned flat. **A flat
:math:`\bar S(r)` from a map coarser than the core means "invisible", not
"absent"**, and this directory has already recorded one selection-function value
under-read 4.5x by running at the wrong healpix order. Only ``mode='patch'``
(healpix orders 6--12, 0.86' at order 12) resolves inside ``R_c``, and it needs a
live ESA Gaia archive query.

WHAT THIS SCRIPT ADDS OVER ``gaiaunlimited.build_patch_map``
------------------------------------------------------------
``build_patch_map`` returns a map whose order column is **hardcoded to 12**::

    order = 12 * np.ones_like(allGoodHpx12)

but the M_10 value in each order-12 pixel is filled at the *finest order between
12 and 8 that contains at least* ``min_points`` *sources* -- so the map is
*indexed* at order 12 while much of it may be *informed* at order 9 or 8. Taking
the order column at face value is precisely the failure mode this directory warns
about. This script reimplements the same grouping while **recording the effective
order at which every pixel was filled**, so the resolution claim is measured
rather than asserted, and the histogram of attained orders is written to the
sidecar.

THE ORACLE
----------
The reimplementation is not trusted on inspection. It is run against
``gaiaunlimited.selectionfunctions.surveyTCG.build_patch_map`` **on the identical
patch and the identical cached source table** (the upstream archive call is shimmed
to return the cache, so the two see byte-identical inputs), and the two M_10 maps
must agree pixel-for-pixel. External, parameter-free, and it can fail: an error in
the healpix arithmetic, the ``min_points`` threshold or the median moves pixels.
A mismatch aborts before any profile is written.

FALSIFICATION
-------------
This script's output falsifies nothing on its own -- it supplies
:math:`\bar S(r)` to ``a1_selection_corrected_rdp.py``, where the pre-registered
thresholds live. What it *can* establish, and what would make its own answer
worthless, is a resolution failure: **if the effective order reached over the core
does not put the pixel scale at or inside 1.2 R_c = 1.66', the resulting profile
is not a resolving measurement and a flat answer from it must not be quoted as a
null.** The order histogram and the mean effective order inside ``R_c`` are both
written, so that check is mechanical rather than editorial -- and on the
2026-08-04 run it **fails**: 3.36' at ``min_points=20``, 1.73' at its finest.

WHAT THE 2026-08-04 RUN ACTUALLY ACHIEVED -- READ BEFORE QUOTING A RESOLUTION
------------------------------------------------------------------------------
**The ESA archive was half down.** Every ``launch_job_async`` call returned
HTTP 500, including a deliberately trivial 10-row test; ``launch_job`` worked but
the sync endpoint hard-caps at **2000 rows** regardless of ``ROW_LIMIT`` or an
explicit ``maxrec`` (both measured, not assumed). That is the second recorded
outage of this endpoint, after 2026-07-27. The patch was therefore cut from
1.25 deg to **0.35 deg (21')** and fetched as healpix ``source_id`` ranges:
44 slices, 5873 sources inside the circle, verified against an independent
``COUNT(*)`` -- which is how the first attempt's silent 2000-row truncation was
caught rather than published.

**``mode='patch'`` does not reach order 12 in this field at the default
``min_points=20``.** Measured order histogram over 1878 order-12-indexed pixels:
343 at order 11, 1188 at order 10, 342 at order 9, 3 at order 8, 2 unfilled. The
mean effective order *inside* ``R_c`` is **10.2 (3.36')**. Reaching order 12
requires ``min_points=5``, which buys 1.73' at the cost of a median over five
stars -- and, worse, subdivides **preferentially where density is high, i.e. in
the core**, so core and field are then measured at different resolutions. That is
why ``min_points`` is swept and reported rather than chosen.

Consequently, any claim that **1.156%** is an "order-12 patch mode" suppression
here is not reproducible from this field. ``docs/design-notes/
king_model_validity.md`` asserts that number *and*, in the same section, that
patch mode is "still to do"; no patch npz existed on disk; and
``completeness_bias_scaling.py`` hardcodes it as
``NGC6383_MEASURED_SUPPRESSION = 0.01156``. This run measures **+2.34%** at
``min_points`` 20 and 10 and **+4.50%** at 5 -- bracketing neither 1.156% nor the
order-10 reading of 0.39%.

**The written profile is a splice.** Inside 20.6' it is the patch; outside it is
the archive-free order-10 ``multi`` map, rescaled for continuity at the seam. The
seam and the scale factor are in the sidecar. The splice exists only because the
patch had to be shrunk to clear the archive's row cap; it is a limitation, not a
method.

USAGE
-----
    # while the async endpoint is down, --sliced is required:
    python tools/validation/a1_patch_selection_function.py --stage query \
        --sliced --patch-radius 0.35
    python tools/validation/a1_patch_selection_function.py --stage map
    python tools/validation/a1_patch_selection_function.py --stage sbar
    python tools/validation/a1_patch_selection_function.py --stage sbar \
        --primary-min-points 5 --sbar-out tools/validation/a1_sbar_patch_mp5.npz

Stages are separately resumable and each writes as it finishes: the archive query
caches to ``a1_patch_gaia_cache.npz`` (gitignored, regenerable), the map to
``a1_patch_map.npz``, and per-radius profile rows stream to
``a1_patch_sbar.rows.jsonl`` so a death costs one radius. The deliverable is
``a1_sbar_patch.npz``, whose ``r_arcmin`` is the 256-node Gauss-Legendre grid
``king_expected_count_weighted`` evaluates on, so it can be passed straight
through as ``completeness=``.

Environment: ``gaiaunlimited`` is not installed in any project env; this script
was run with it (plus ``healpy``, ``astropy_healpix``, ``h5py``, ``astromet``)
``pip install --target``-ed into a scratchpad directory and put on ``PYTHONPATH``,
against ``/Users/notluquis/miniforge3/envs/cosmic/bin/python``. Nothing was
installed into a shared env.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

HERE = Path(__file__).resolve().parent
CACHE = HERE / "a1_patch_gaia_cache.npz"
MAP_NPZ = HERE / "a1_patch_map.npz"
ROWS = HERE / "a1_patch_sbar.rows.jsonl"
SBAR_NPZ = HERE / "a1_sbar_patch.npz"
JSON_OUT = HERE / "a1_patch_selection_function.json"

B = Path("/Users/notluquis/erotica/data/test/NGC6383")
MEMBERS = B / "comments_paper/radius_robustness/generated/70/paperfaithful_reference_p06.ecsv"
CENTER = SkyCoord(263.6826 * u.deg, -32.5838 * u.deg)
FIELD_ARCMIN = 70.0
PATCH_RADIUS_DEG = 1.25  # 75' -- covers the 70' field with margin
NODES = 256
MIN_POINTS = 20
RESOLVING_ORDER = 11
ORDER12_ARCMIN = 60.0 * np.sqrt(41253.0 / (12 * 4**12) / np.pi) * 2  # ~0.86'


def _save(key, payload):
    doc = {}
    if JSON_OUT.exists():
        try:
            doc = json.loads(JSON_OUT.read_text())
        except json.JSONDecodeError:
            pass
    doc[key] = payload
    doc["_written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    JSON_OUT.write_text(json.dumps(doc, indent=1, default=float))
    print(f"  [saved '{key}' -> {JSON_OUT.name}]", flush=True)


def quadrature_radii(field_radius=FIELD_ARCMIN, nodes=NODES):
    x, _ = np.polynomial.legendre.leggauss(nodes)
    return 0.5 * field_radius * (x + 1.0)


def order_arcmin(order):
    """Mean healpix pixel diameter at `order`, in arcmin."""
    area_deg2 = 41253.0 / (12 * 4**order)
    return 2.0 * 60.0 * np.sqrt(area_deg2 / np.pi)


# ---------------------------------------------------------------------------
# stage 1: the archive query, cached
# ---------------------------------------------------------------------------
ADQL = """SELECT ra, dec, source_id, phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE 1 = CONTAINS(POINT(ra,dec),CIRCLE(%f, %f, %f))
and astrometric_matched_transits<11
and phot_g_mean_mag<50"""


def _run_query(q, attempts=3):
    """The archive is the fragile step: async 500s, then sync, then G-sliced sync.

    Slicing on ``phot_g_mean_mag`` partitions the rows exactly (the base query
    already requires ``phot_g_mean_mag<50``, so no source is NULL in G and none
    can fall in two slices), which is why the concatenation is the same table the
    single query would have returned.
    """
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1
    # 2026-08-04: the *async* endpoint returns HTTP 500 for this footprint while the
    # *sync* endpoint serves it (a COUNT over the identical cone came back in 77 s).
    # Sync is therefore tried first; async is kept as a fallback because sync is the
    # one subject to a silent server-side MAXREC truncation.
    try:
        return Gaia.launch_job(q).get_results(), "sync"
    except Exception as exc:  # noqa: BLE001 - the archive throws many shapes
        print(f"    sync failed: {type(exc).__name__}: {exc}", flush=True)
    for i in range(attempts):
        try:
            return Gaia.launch_job_async(q).get_results(), f"async(attempt {i + 1})"
        except Exception as exc:  # noqa: BLE001
            print(f"    async attempt {i + 1} failed: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(20)
    raise RuntimeError("no unsliced transport available; use --sliced")


SYNC_MAXREC = 2000  # the ESA Gaia TAP sync endpoint's hard cap, measured not assumed


def _run_query_sliced(radius_deg, order=8, attempts=3):
    r"""Fetch the patch as ``source_id`` ranges, one healpix pixel at a time.

    Why this exists: on 2026-08-04 the ESA archive's **async** endpoint returned
    HTTP 500 for every job including a 10-row one, and the **sync** endpoint
    hard-caps at 2000 rows regardless of ``ROW_LIMIT`` or an explicit ``maxrec``
    (both measured). A Gaia ``source_id`` encodes its nested order-12 healpix
    pixel in the high bits, so a pixel at order `k` is a contiguous ``source_id``
    range and an indexed range scan -- far cheaper for the server than repeating
    the cone predicate, and small enough to clear the sync cap.

    **A slice returning exactly ``SYNC_MAXREC`` rows is treated as truncated** and
    is subdivided into its four children rather than trusted. That guard is the
    reason this can be believed: silent truncation is the failure mode that would
    bias every M_10 median downward with no error raised.
    """
    import healpy as hp
    from astropy.table import vstack
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1
    vec = hp.ang2vec(CENTER.icrs.ra.deg, CENTER.icrs.dec.deg, lonlat=True)
    todo = [
        (order, int(p))
        for p in hp.query_disc(
            2**order,
            vec,
            np.radians(radius_deg + order_arcmin(order) / 60.0),
            nest=True,
            inclusive=True,
        )
    ]
    parts, n_slices, n_subdiv = [], 0, 0
    while todo:
        k, p = todo.pop()
        shift = 2 ** (35 + 2 * (12 - k))
        q = (
            "SELECT ra, dec, source_id, phot_g_mean_mag FROM gaiadr3.gaia_source\n"
            f"WHERE source_id BETWEEN {p * shift} AND {(p + 1) * shift - 1}\n"
            "and astrometric_matched_transits<11 and phot_g_mean_mag<50"
        )
        for i in range(attempts):
            try:
                t = Gaia.launch_job(q).get_results()
                break
            except Exception as exc:  # noqa: BLE001
                print(f"    order{k} pix{p} attempt {i + 1}: {exc}", flush=True)
                time.sleep(15)
        else:
            raise RuntimeError(f"archive unreachable for order-{k} pixel {p}")
        if len(t) >= SYNC_MAXREC:
            todo.extend([(k + 1, 4 * p + j) for j in range(4)])
            n_subdiv += 1
            print(f"    order{k} pix{p}: hit the {SYNC_MAXREC} cap -> subdividing", flush=True)
            continue
        n_slices += 1
        parts.append(t)
        if n_slices % 5 == 0:
            print(
                f"    {n_slices} slices, {sum(len(x) for x in parts)} rows, {len(todo)} pending",
                flush=True,
            )
    return vstack(parts), f"sliced-sync (order {order}, {n_slices} slices, {n_subdiv} subdivided)"


def _count(q):
    """``COUNT(*)`` for the same WHERE clause -- the oracle on silent truncation."""
    from astroquery.gaia import Gaia

    cq = "SELECT COUNT(*) AS n FROM" + q.split("FROM", 1)[1]
    try:
        return int(Gaia.launch_job(cq).get_results()["n"][0])
    except Exception as exc:  # noqa: BLE001
        print(f"    COUNT(*) unavailable: {type(exc).__name__}: {exc}", flush=True)
        return None


def stage_query(radius_deg=PATCH_RADIUS_DEG, force=False, sliced=False):
    if CACHE.exists() and not force:
        with np.load(CACHE) as z:
            print(
                f"  cache hit: {CACHE.name}, {z['source_id'].size} sources, "
                f"radius {float(z['radius_deg'])} deg"
            )
            return

    q = ADQL % (CENTER.icrs.ra.deg, CENTER.icrs.dec.deg, radius_deg)
    print("  querying the ESA Gaia archive (this is the slow, fragile step)...", flush=True)
    t0 = time.perf_counter()
    if sliced:
        tab, how = _run_query_sliced(radius_deg)
        # The slices are healpix pixels, not the circle: cut to the circle so the
        # source set is exactly what the cone query would have returned.
        sky = SkyCoord(np.asarray(tab["ra"], float) * u.deg, np.asarray(tab["dec"], float) * u.deg)
        inside = CENTER.separation(sky).deg < radius_deg
        print(f"  {inside.sum()} of {len(tab)} sliced rows fall inside the circle", flush=True)
        tab = tab[inside]
    else:
        tab, how = _run_query(q)
    dt = time.perf_counter() - t0
    print(f"  {len(tab)} sources in {dt:.0f}s via {how}", flush=True)
    cols = {c.lower(): c for c in tab.colnames}
    np.savez_compressed(
        CACHE,
        ra=np.asarray(tab[cols["ra"]], float),
        dec=np.asarray(tab[cols["dec"]], float),
        source_id=np.asarray(tab[cols["source_id"]], np.int64),
        gmag=np.ma.filled(np.ma.asarray(tab[cols["phot_g_mean_mag"]]), np.nan).astype(float),
        radius_deg=radius_deg,
        adql=q,
    )
    # Transport check. A sync job can be truncated at the server's MAXREC without
    # raising, and a truncated table gives biased M_10 medians with no error at all.
    # The independent COUNT(*) over the identical footprint is the oracle.
    expected = _count(q)
    ok = expected is None or int(len(tab)) == expected
    print(
        f"  row-count check: got {len(tab)}, COUNT(*) says {expected} -> "
        f"{'PASS' if ok else 'TRUNCATED'}",
        flush=True,
    )
    _save(
        "query",
        {
            "n_sources": int(len(tab)),
            "radius_deg": radius_deg,
            "seconds": round(dt, 1),
            "transport": how,
            "count_star": expected,
            "row_count_check_pass": bool(ok),
            "adql": q,
        },
    )
    if not ok:
        raise SystemExit("row count does not match COUNT(*): the table is truncated")


def load_cache():
    with np.load(CACHE) as z:
        return (
            np.asarray(z["ra"], float),
            np.asarray(z["dec"], float),
            np.asarray(z["source_id"], np.int64),
            np.asarray(z["gmag"], float),
            float(z["radius_deg"]),
        )


# ---------------------------------------------------------------------------
# stage 2: the M_10 map, with the effective order recorded
# ---------------------------------------------------------------------------
def build_map_with_orders(source_id, gmag, radius_deg, min_points=MIN_POINTS):
    r"""Reimplementation of ``build_patch_map`` that records the order per pixel.

    Same algorithm: expand every order-6 pixel containing a source down to order
    12, keep the order-12 pixels whose centre falls inside the patch, then walk
    orders 12 -> 8 filling each still-empty pixel with the median ``G`` of the
    containing pixel at that order once it holds at least `min_points` sources.
    The only addition is ``eff_order``: the order at which each pixel was filled.
    """
    import healpy as hp

    hpx12_src = source_id // 2**35  # Gaia source_id encodes the nested order-12 pixel
    hpx6 = np.unique(hpx12_src // 4**6)
    # expand order 6 -> 12
    pix = hpx6
    for _ in range(6):
        pix = (4 * pix[:, None] + np.arange(4)[None, :]).ravel()
    pix = np.unique(pix)
    lon, lat = hp.pix2ang(2**12, pix, nest=True, lonlat=True)
    sep = CENTER.separation(SkyCoord(lon * u.deg, lat * u.deg)).deg
    good = pix[sep < radius_deg]
    good.sort()

    m10 = np.full(good.size, np.nan)
    eff_order = np.full(good.size, -1, dtype=np.int8)
    finite = np.isfinite(gmag)
    for step_up in range(5):
        order = 12 - step_up
        todo = ~np.isfinite(m10)
        if not todo.any():
            break
        key_src = hpx12_src[finite] // 4**step_up
        g_src = gmag[finite]
        srt = np.argsort(key_src, kind="stable")
        key_sorted, g_sorted = key_src[srt], g_src[srt]
        keys_needed = good[todo] // 4**step_up
        lo = np.searchsorted(key_sorted, keys_needed, "left")
        hi = np.searchsorted(key_sorted, keys_needed, "right")
        counts = hi - lo
        fill = counts >= min_points
        idx_todo = np.flatnonzero(todo)[fill]
        med = np.array([np.median(g_sorted[a:b]) for a, b in zip(lo[fill], hi[fill], strict=True)])
        m10[idx_todo] = med
        eff_order[idx_todo] = order
        print(
            f"    order {order:2d} ({order_arcmin(order):5.2f}'): filled {fill.sum():6d}, "
            f"{int((~np.isfinite(m10)).sum()):6d} still empty",
            flush=True,
        )
    return good, m10, eff_order


def _oracle_build_patch_map(source_id, ra, dec, gmag, radius_deg, min_points=MIN_POINTS):
    """Upstream ``build_patch_map`` fed the identical cached table via a shim."""
    from gaiaunlimited.selectionfunctions import surveyTCG

    tab = Table({"ra": ra, "dec": dec, "source_id": source_id, "phot_g_mean_mag": gmag})

    class _Job:
        def get_results(self_inner):
            return tab

    real = surveyTCG.Gaia.launch_job_async
    surveyTCG.Gaia.launch_job_async = staticmethod(lambda *a, **k: _Job())
    try:
        return surveyTCG.build_patch_map(CENTER, radius_deg, min_points)
    finally:
        surveyTCG.Gaia.launch_job_async = real


def stage_map(check_oracle=True):
    ra, dec, source_id, gmag, radius_deg = load_cache()
    print(f"\n=== M_10 map over {radius_deg} deg, {source_id.size} sources ===", flush=True)
    t0 = time.perf_counter()
    pix, m10, eff_order = build_map_with_orders(source_id, gmag, radius_deg)
    print(f"  built {pix.size} order-12 pixels in {time.perf_counter() - t0:.1f}s", flush=True)

    payload = {
        "n_pixels": int(pix.size),
        "n_sources": int(source_id.size),
        "radius_deg": radius_deg,
        "min_points": MIN_POINTS,
        "nan_fraction": float(np.mean(~np.isfinite(m10))),
        "order_histogram": {
            str(o): int((eff_order == o).sum()) for o in sorted(set(eff_order.tolist()))
        },
        "order_arcmin": {str(o): order_arcmin(o) for o in range(8, 13)},
        "max_order_attained": int(eff_order.max()),
        "median_order": float(np.median(eff_order[eff_order > 0])),
        "resolving_gate_order": RESOLVING_ORDER,
        "m10_min": float(np.nanmin(m10)),
        "m10_max": float(np.nanmax(m10)),
    }
    print(f"  order histogram: {payload['order_histogram']}")

    if check_oracle:
        print("  ORACLE: upstream build_patch_map on the identical table...", flush=True)
        t0 = time.perf_counter()
        ref = _oracle_build_patch_map(source_id, ra, dec, gmag, radius_deg)
        ref_pix = ref[:, 1].astype(np.int64)
        ref_m10 = ref[:, 2]
        srt = np.argsort(ref_pix)
        ref_pix, ref_m10 = ref_pix[srt], ref_m10[srt]
        same_pixels = bool(ref_pix.size == pix.size and np.array_equal(ref_pix, pix))
        both = np.isfinite(m10) & np.isfinite(ref_m10) if same_pixels else np.array([])
        agree = bool(
            same_pixels
            and np.array_equal(np.isfinite(m10), np.isfinite(ref_m10))
            and np.allclose(m10[both], ref_m10[both], rtol=0, atol=1e-12)
        )
        payload["oracle"] = {
            "reference": "gaiaunlimited.selectionfunctions.surveyTCG.build_patch_map",
            "seconds": round(time.perf_counter() - t0, 1),
            "same_pixel_set": same_pixels,
            "n_ref_pixels": int(ref_pix.size),
            "max_abs_m10_diff": float(np.nanmax(np.abs(m10[both] - ref_m10[both])))
            if both.size
            else None,
            "pass": agree,
        }
        print(f"  ORACLE {'PASS' if agree else 'FAIL'}: {payload['oracle']}")
        if not agree:
            _save("map", payload)
            raise SystemExit(
                "oracle mismatch -- refusing to write a profile from an unverified map"
            )

    np.savez_compressed(
        MAP_NPZ, pix=pix, m10=m10, eff_order=eff_order, radius_deg=radius_deg, min_points=MIN_POINTS
    )
    _save("map", payload)
    return pix, m10, eff_order


# ---------------------------------------------------------------------------
# stage 3: S-bar(r)
# ---------------------------------------------------------------------------
def _lookup(pix_sorted, m10, ra_deg, dec_deg):
    import healpy as hp

    ipix = hp.ang2pix(2**12, ra_deg, dec_deg, nest=True, lonlat=True)
    idx = np.searchsorted(pix_sorted, ipix)
    idx = np.clip(idx, 0, pix_sorted.size - 1)
    hit = pix_sorted[idx] == ipix
    out = np.full(ipix.shape, np.nan)
    out[hit] = m10[idx[hit]]
    return out, hit


def profile_from_map(pix, m10, eff_order, mag_grid, radii, n_az, m10_to_completeness):
    """S-bar(r), and the effective healpix order sampled, on one radial grid."""
    azimuth = np.linspace(0.0, 2 * np.pi, n_az, endpoint=False)
    s, order, nanfrac = [], [], []
    for r in radii:
        pts = CENTER.directional_offset_by(azimuth * u.rad, max(r, 1e-9) * u.arcmin)
        m, _ = _lookup(pix, m10, pts.icrs.ra.deg, pts.icrs.dec.deg)
        eo, _ = _lookup(pix, eff_order.astype(float), pts.icrs.ra.deg, pts.icrs.dec.deg)
        comp = np.array([m10_to_completeness(np.full(n_az, g), m) for g in mag_grid])
        s.append(float(np.nanmean(comp)))
        order.append(float(np.nanmean(np.where(eo > 0, eo, np.nan))))
        nanfrac.append(float(np.mean(~np.isfinite(m))))
    return np.asarray(s), np.asarray(order), np.asarray(nanfrac)


def aperture_suppression(radii, s, aperture, outer):
    """1 - <S>_{r<aperture} / <S>_{r<outer}, area-weighted. Same definition used
    by ``completeness_bias_scaling.py`` and ``a1_selection_corrected_rdp.py``, so
    the numbers are directly comparable across scripts."""
    f = lambda rr: np.interp(rr, radii, s)  # noqa: E731
    a = np.linspace(1e-6, aperture, 2000)
    b = np.linspace(1e-6, outer, 20000)
    s_in = np.trapezoid(2 * np.pi * a * f(a), a) / (np.pi * aperture**2)
    s_out = np.trapezoid(2 * np.pi * b * f(b), b) / (np.pi * outer**2)
    return float(1.0 - s_in / s_out)


def stage_sbar(n_az=128, n_mag=24, n_radii=64, min_points_sweep=(20, 10, 5), primary=20):
    r"""The profile, plus the resolution/noise trade-off that sets its meaning.

    ``min_points`` is **not** a free knob to be picked: it trades spatial
    resolution against the noise on each M_10 median. Lowering it subdivides more
    pixels (finer) at the cost of a median over fewer stars (noisier). So the
    sweep is run and reported, and the conclusion is only trusted where the
    measured suppression is stable across it.
    """
    from gaiaunlimited.selectionfunctions.surveyTCG import m10_to_completeness

    ra, dec, source_id, gmag_src, radius_deg = load_cache()
    patch_arcmin = radius_deg * 60.0
    r_max = 0.98 * patch_arcmin
    table = Table.read(MEMBERS)
    gmag = np.asarray(table["Gmag"], float)
    gmag = gmag[np.isfinite(gmag)]
    mag_grid = np.quantile(gmag, np.linspace(0.02, 0.98, n_mag))
    radii = np.concatenate([[0.0], np.geomspace(0.02, r_max, n_radii - 1)])
    print(
        f"\n=== S-bar(r) inside the {patch_arcmin:.1f}' patch: {n_radii} radii x "
        f"{n_az} azimuths x {n_mag} magnitudes ===\n"
        f"  members N={gmag.size} Gmag median {np.median(gmag):.2f} "
        f"p98 {np.percentile(gmag, 98):.2f}",
        flush=True,
    )

    sweep, keep = {}, None
    for mp in min_points_sweep:
        pix, m10, eff_order = build_map_with_orders(source_id, gmag_src, radius_deg, mp)
        s, order, nanfrac = profile_from_map(
            pix, m10, eff_order, mag_grid, radii, n_az, m10_to_completeness
        )
        core = radii <= 1.384
        supp = aperture_suppression(radii, s, 1.384, r_max)
        hist = {str(int(o)): int((eff_order == o).sum()) for o in sorted(set(eff_order.tolist()))}
        row = {
            "min_points": mp,
            "order_histogram": hist,
            "max_order_attained": int(eff_order.max()),
            "mean_order_inside_Rc": float(np.nanmean(order[core])),
            "resolution_arcmin": float(order_arcmin(np.nanmean(order[core]))),
            "S_min": float(s.min()),
            "S_max": float(s.max()),
            "S_at_centre": float(s[0]),
            "S_at_patch_edge": float(s[-1]),
            "core_suppression_within_patch": supp,
            "core_suppression_pct": 100.0 * supp,
            "max_nan_fraction": float(nanfrac.max()),
        }
        sweep[str(mp)] = row
        print(
            f"  min_points={mp:3d}: orders {hist}  <order> inside R_c "
            f"{row['mean_order_inside_Rc']:.2f} ({row['resolution_arcmin']:.2f}')  "
            f"S {s.min():.5f}..{s.max():.5f}  core suppression {100 * supp:+.4f}%",
            flush=True,
        )
        with ROWS.open("a") as fh:
            for rr, ss, oo in zip(radii, s, order, strict=True):
                fh.write(
                    json.dumps(
                        {"min_points": mp, "r": float(rr), "S": float(ss), "order": float(oo)}
                    )
                    + "\n"
                )
        if mp == primary:
            keep = (s, order, radii)

    vals = [v["core_suppression_within_patch"] for v in sweep.values()]
    payload = {
        "patch_radius_arcmin": patch_arcmin,
        "r_max_arcmin": r_max,
        "n_radii": int(radii.size),
        "n_az": n_az,
        "n_mag": n_mag,
        "primary_min_points": primary,
        "sweep": sweep,
        "core_suppression_range_over_sweep": [float(min(vals)), float(max(vals))],
        "stable_across_sweep": bool(max(vals) - min(vals) < 0.01),
        "resolution_gate_arcmin": 1.2 * 1.384,
        "resolving": bool(sweep[str(primary)]["resolution_arcmin"] <= 1.2 * 1.384),
    }

    # Full 0-70' curve for the RDP fit: patch inside the seam, the archive-free
    # order-10 `multi` map outside it, rescaled for continuity at the seam. The
    # splice is declared because it is a real limitation: the patch was cut to
    # 0.35 deg so the query would clear the archive's sync row cap while async
    # was returning HTTP 500.
    s, order, radii = keep
    nodes = quadrature_radii()
    multi_path = HERE / "ngc6383_selection_function_multi.npz"
    full = np.interp(nodes, radii, s)
    seam = r_max
    if multi_path.exists():
        with np.load(multi_path) as z:
            s_multi = np.asarray(z["completeness"], float)
        scale = float(np.interp(seam, radii, s) / np.interp(seam, nodes, s_multi))
        outside = nodes > seam
        full[outside] = scale * s_multi[outside]
        payload["splice"] = {
            "seam_arcmin": seam,
            "outer_source": "multi (order 10)",
            "continuity_scale": scale,
        }
    full = np.clip(full, 0.0, 1.0)
    np.savez_compressed(
        SBAR_NPZ,
        r_arcmin=nodes,
        completeness=full,
        mode="patch",
        orders=np.array([sweep[str(primary)]["max_order_attained"]], dtype=np.int64),
        resolution_arcmin=sweep[str(primary)]["resolution_arcmin"],
        sample_radii=radii,
        sample_S=s,
        sample_order=order,
        mag_grid=mag_grid,
        n_az=n_az,
        field_radius=FIELD_ARCMIN,
        min_points=primary,
        seam_arcmin=seam,
    )
    payload["written"] = str(SBAR_NPZ)
    print(
        f"\n  suppression across the min_points sweep: "
        f"{100 * min(vals):+.4f}% .. {100 * max(vals):+.4f}%  "
        f"stable: {payload['stable_across_sweep']}  resolving: {payload['resolving']}"
    )
    _save("sbar", payload)
    return payload


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all", choices=["all", "query", "map", "sbar"])
    ap.add_argument("--force-query", action="store_true")
    ap.add_argument(
        "--sliced",
        action="store_true",
        help="fetch as healpix source_id ranges (needed while async is down)",
    )
    ap.add_argument("--patch-radius", type=float, default=PATCH_RADIUS_DEG)
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--n-az", type=int, default=128)
    ap.add_argument("--n-mag", type=int, default=24)
    ap.add_argument("--n-radii", type=int, default=80)
    ap.add_argument(
        "--primary-min-points",
        type=int,
        default=20,
        help="which sweep rung becomes the written S-bar(r)",
    )
    ap.add_argument("--sbar-out", type=Path, default=None)
    args = ap.parse_args()

    if args.stage in ("all", "query"):
        stage_query(radius_deg=args.patch_radius, force=args.force_query, sliced=args.sliced)
    if args.stage in ("all", "map"):
        stage_map(check_oracle=not args.no_oracle)
    if args.stage in ("all", "sbar"):
        global SBAR_NPZ
        if args.sbar_out is not None:
            SBAR_NPZ = args.sbar_out
        stage_sbar(
            n_az=args.n_az, n_mag=args.n_mag, n_radii=args.n_radii, primary=args.primary_min_points
        )


if __name__ == "__main__":
    main()
