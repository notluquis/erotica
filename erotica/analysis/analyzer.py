"""High-level cluster analysis utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from astropy.table import QTable

from erotica.core.clustering import Clustering

from .._membership import COLUMNA_ISOCRONA, COLUMNA_POR_DEFECTO, select_by_probability
from ._clipping import DEFAULT_MAXITERS as _CLIP_MAXITERS
from ._clipping import sigma_clip_parallax as _sigma_clip_parallax
from ._io import _DEPRECATED, _warn_dill_cache, load_dataset
from ._isochrone import IsochroneFitter
from ._plots import plot_persistence_vs_members, plot_pms_panels, plot_probability_vs_gmag
from ._sagitta import pms_characterization as _pms_characterization
from .figures import plot_cumulative as _plot_cumulative
from .figures import plot_distance_pathway_overlap as _plot_distance_pathway_overlap
from .kinematics import pm_amplitude as _pm_amplitude
from .structure import calculate_half_light_radius as _calculate_half_light_radius
from .structure import center_determination as _center_determination
from .structure import half_mass_radius as _half_mass_radius


class ClusterAnalyzer:
    """High-level helper around :class:`~clustering.Clustering` and plotting utilities."""

    def __init__(
        self,
        file_obj,
        *,
        dataloader_kwargs: dict | None = None,
        dill_cache: Any = _DEPRECATED,
        search_method: str = "optuna",
        output_dir: str | None = None,
        verbose: int = logging.INFO,
        debug_mode: bool = False,
    ) -> None:
        """Load the dataset and set up output paths.

        Unlike the other facades in this package, this constructor **does
        work**: it resolves `file_obj`, reads the catalogue, and may restore a
        previously pickled clustering run.

        Parameters
        ----------
        file_obj : str or DataFrame or Table
            The input catalogue, dispatched on type. A ``str`` is treated as a
            path; a :class:`pandas.DataFrame` or an Astropy ``Table``/``QTable``
            is adopted in memory. Anything else raises ``TypeError``.

            A path ending in ``.dill`` is unpickled rather than read: a payload
            that is a :class:`~erotica.core.Clustering` restores ``self.data``
            **and** ``self.clustering``, while a dict payload (keyed ``data``,
            ``table`` or ``qtable``) restores only the table. Any other path is
            read through :class:`~erotica.io.DataLoader`.
        dataloader_kwargs : dict, optional
            Forwarded to :meth:`~erotica.io.DataLoader.load_data` -- the column
            selection knobs (``systems``, ``include_distances``, ``fidelity``
            and so on). Ignored by the in-memory and ``.dill`` paths, which have
            nothing to load.
        dill_cache : deprecated
            **Accepted and ignored; passing it raises ``DeprecationWarning``.**
            Removed 2026-08-04 because it was never a cache. It wrote a dict
            payload beside the input with a ``.dill`` suffix, but the read path
            fired only when the *caller's own* `file_obj` already ended in
            ``.dill``, so the sidecar it wrote was never consulted and every run
            re-read the source. Nothing in the package or the test suite ever
            read the resulting ``dill_path`` attribute, which is also gone.

            It was deleted rather than repaired: a working cache needs an mtime
            check against the source and a comparison of `dataloader_kwargs`,
            neither of which exists anywhere here, and without them a stale
            sidecar silently shadowing an edited catalogue is a worse bug than
            the redundant read. To load a pickle, pass the ``.dill`` path itself
            as `file_obj`; to write one, call :func:`dill.dump` yourself, which
            is what ``tools/validation/ngc6383_radius_robustness.py`` does.
        search_method : str, default "optuna"
            Stored for the :class:`~erotica.core.Clustering` instances this
            object builds on demand, notably in :meth:`clusters_summary` when no
            clustering was restored. It starts no search by itself.
        output_dir : str, optional
            Where derived files are written. Defaults to the input file's own
            directory for a path input, and is created if missing.
        verbose : int, default ``logging.INFO``
            Logging level handed to the loader.
        debug_mode : bool, default False
            Force ``DEBUG`` logging and Astropy's debug records; see
            :class:`~erotica.io.DataLoader`.

        Raises
        ------
        TypeError
            If `file_obj` is none of the accepted types, or if a ``.dill``
            payload unpickles to an unsupported type.
        KeyError
            If a dict ``.dill`` payload has no recognised table key.

        Notes
        -----
        ``selected_cluster`` starts as ``None``; several plotting methods
        require it and raise ``ValueError`` asking for ``select_cluster()``
        rather than guessing a cluster.

        ``self.clustering`` is ``None`` for every input form except a ``.dill``
        holding a pickled ``Clustering``.
        """
        _warn_dill_cache(dill_cache)
        self.selected_cluster: int | None = None
        self.search_method = search_method

        result = load_dataset(
            file_obj,
            dataloader_kwargs=dataloader_kwargs,
            output_dir=output_dir,
            verbose=verbose,
            debug_mode=debug_mode,
        )

        self.data: QTable = result["data"]
        self.loader = result["loader"]
        self.clustering: Clustering | None = result["clustering"]
        self.base_dir = Path(result["base_dir"])
        self.output_dir = Path(result["output_dir"])
        self.source = result["source"]

    @property
    def dill_path(self):
        """Removed with the write-only ``dill_cache``. Raises with an explanation.

        The ``dill_cache`` keyword kept a deprecation shim because this is a released v0.1.0
        class; the attribute beside it was deleted outright, so the same call gave a courteous
        warning for the keyword and a bare ``AttributeError`` for the attribute. This equalises
        that: still an ``AttributeError``, but one that says what happened and what to do.
        """
        raise AttributeError(
            "ClusterAnalyzer.dill_path was removed together with the dill_cache sidecar, which "
            "was written on every load and never read back. Nothing consumed this attribute. "
            "If you need a cached copy of the loaded table, write it yourself from "
            "`analyzer.data`."
        )

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def clusters_summary(self, **kwargs) -> pd.DataFrame:
        if self.clustering is not None:
            return self.clustering.get_cluster_summary(**kwargs)
        tmp = Clustering(self.data, search_method=self.search_method)
        return tmp.get_cluster_summary(**kwargs)

    # ------------------------------------------------------------------
    # Plotting wrappers
    # ------------------------------------------------------------------
    def plot_persistence_vs_members(
        self,
        *,
        percentile: float = 0.8,
        figsize: tuple[int, int] = (10, 10),
        save: bool = False,
        filename: str | None = None,
        **plot_kwargs,
    ) -> None:
        summary = self.clusters_summary()
        plot_persistence_vs_members(
            self.data,
            summary,
            percentile=percentile,
            output_dir=self.output_dir,
            save=save,
            filename=filename,
            figsize=figsize,
            plot_kwargs=plot_kwargs,
        )

    def plot_probability_vs_gmag(
        self,
        *,
        cluster: int | None = None,
        prob_thresh: float = 0.6,
        gmag_limit: float | None = None,
        figsize: tuple[int, int] = (7, 6),
        save: bool = False,
        filename: str | None = None,
        **plot_kwargs,
    ) -> None:
        cid = cluster if cluster is not None else self.selected_cluster
        if cid is None:
            raise ValueError(
                "No cluster specified; call select_cluster() first or pass cluster argument."
            )
        self.selected_cluster = cid
        plot_probability_vs_gmag(
            self.data,
            cluster=cid,
            probability_threshold=prob_thresh,
            gmag_limit=gmag_limit,
            figsize=figsize,
            output_dir=self.output_dir,
            save=save,
            filename=filename,
            plot_kwargs=plot_kwargs,
        )

    def plot_pms(
        self,
        *,
        cluster: int | None = None,
        pms_threshold: float = 0.6,
        figsize: tuple[int, int] = (7, 7),
        layout: str = "tight",
        save: bool = False,
        filename: str | None = None,
        age_xlim: tuple[float, float] | None = None,
        av_xlim: tuple[float, float] | None = None,
    ) -> None:
        cid = cluster if cluster is not None else self.selected_cluster
        if cid is None:
            raise ValueError("No cluster specified; call select_cluster() first or pass `cluster`.")
        self.selected_cluster = cid
        plot_pms_panels(
            self.data,
            cluster=cid,
            pms_threshold=pms_threshold,
            figsize=figsize,
            layout=layout,
            output_dir=self.output_dir,
            save=save,
            filename=filename,
            age_xlim=age_xlim,
            av_xlim=av_xlim,
        )

    # ------------------------------------------------------------------
    # Data manipulation helpers
    # ------------------------------------------------------------------
    def select_cluster(self, label: int):
        self.selected_cluster = label
        return self.data[self.data["cluster"] == label]

    def sigma_clip_parallax(
        self,
        *,
        cluster: int | None = None,
        sigma: float = 2.0,
        print_results: bool = False,
        use_biweight: bool = True,
        in_place: bool = True,
        mark_label: int = -1,
        return_mask: bool = False,
        preselector_mask=None,
        method: str = "normalised",
        parallax_error_column: str = "parallax_error",
        maxiters: int | None = _CLIP_MAXITERS,
    ):
        """Clip parallax outliers for one cluster.

        Thin pass-through to
        :func:`erotica.analysis._clipping.sigma_clip_parallax`; `method`,
        `parallax_error_column` and `maxiters` are documented there, including
        what each measured and what is still unmeasured.

        The default `method` moved from the raw parallax to the normalised
        residual on 2026-08-02. On the NGC 6383 pre-clip sample the raw clip
        retains 94.2% of the brightest ``Gmag`` quartile against 72.8% of the
        faintest -- a magnitude-dependent selection function, not only an outlier
        test -- against 93.2% / 100.0% for the shipped normalised form.
        """
        cid = cluster if cluster is not None else self.selected_cluster
        if cid is None:
            raise ValueError(
                "No cluster specified; call select_cluster() first or pass cluster argument."
            )
        result = _sigma_clip_parallax(
            self.data,
            cluster=cid,
            sigma=sigma,
            use_biweight=use_biweight,
            in_place=in_place,
            mark_label=mark_label,
            print_results=print_results,
            return_mask=return_mask,
            preselector_mask=preselector_mask,
            method=method,
            parallax_error_column=parallax_error_column,
            maxiters=maxiters,
        )
        return result

    def pms_characterization(
        self,
        *,
        cluster: int | None = None,
        run_cli: bool = False,
        input_dir: str | None = None,
        output_prefix: str | None = None,
        overwrite_inputs: bool = True,
        return_data: bool = False,
    ):
        cid = cluster if cluster is not None else self.selected_cluster
        if cid is None:
            raise ValueError(
                "No cluster specified; call select_cluster() first or pass `cluster` argument."
            )
        joined = _pms_characterization(
            self.data,
            cluster=cid,
            base_dir=self.base_dir,
            input_dir=input_dir,
            output_prefix=output_prefix,
            overwrite_inputs=overwrite_inputs,
            run_cli=run_cli,
            return_data=return_data,
        )
        return joined

    # ------------------------------------------------------------------
    # Migrated paper-analysis conveniences
    # ------------------------------------------------------------------
    def center_determination(
        self, *, cluster: int | None = None, prob_threshold: float | None = None, **kwargs
    ):
        """Estimate the center for the selected cluster using the migrated KDE helper."""
        cid = cluster if cluster is not None else self.selected_cluster
        table = self.data if cid is None else self.data[self.data["cluster"] == cid]
        table = select_by_probability(table, COLUMNA_POR_DEFECTO, prob_threshold)
        return _center_determination(table, **kwargs)

    def half_mass_radius(
        self, center, *, cluster: int | None = None, prob_threshold: float | None = None, **kwargs
    ):
        """Calculate half-mass radius for the selected cluster."""
        cid = cluster if cluster is not None else self.selected_cluster
        table = self.data if cid is None else self.data[self.data["cluster"] == cid]
        table = select_by_probability(table, COLUMNA_POR_DEFECTO, prob_threshold)
        return _half_mass_radius(table, center, **kwargs)

    def half_light_radius(
        self, center, *, cluster: int | None = None, prob_threshold: float | None = None, **kwargs
    ):
        """Calculate half-light radius for the selected cluster."""
        cid = cluster if cluster is not None else self.selected_cluster
        table = self.data if cid is None else self.data[self.data["cluster"] == cid]
        table = select_by_probability(table, COLUMNA_POR_DEFECTO, prob_threshold)
        return _calculate_half_light_radius(table, center, **kwargs)

    def add_pm_amplitude(
        self,
        *,
        pmra_column: str = "pmra",
        pmdec_column: str = "pmdec",
        output_column: str = "projected_velocity",
    ):
        """Annotate ``self.data`` with total proper-motion amplitude."""
        self.data[output_column] = _pm_amplitude(self.data[pmra_column], self.data[pmdec_column])
        return self.data[output_column]

    def plot_cumulative(self, centers, *, filename: str = "integred_magnitude.pdf", **kwargs):
        """Plot cumulative integrated magnitude into ``output_dir`` by default."""
        savefig = kwargs.pop("savefig", self.output_dir / filename)
        return _plot_cumulative(self.data, centers, savefig=savefig, **kwargs)

    def plot_distance_pathway_overlap(
        self, *, filename: str = "distance_pathway_overlap.pdf", **kwargs
    ):
        """Plot distance-pathway comparison into ``output_dir`` by default."""
        savefig = kwargs.pop("savefig", self.output_dir / filename)
        return _plot_distance_pathway_overlap(savefig=savefig, **kwargs)

    # ------------------------------------------------------------------
    # Isochrone fitting
    # ------------------------------------------------------------------

    def prepare_isochrone_fitter(
        self,
        *,
        isochs_path: str | Path,
        cluster: int | None = None,
        prob_threshold: float = 0.6,
        probability_column: str = COLUMNA_ISOCRONA,
        loga_range: tuple[float, float],
        Av_range: tuple[float, float] = (0.0, 3.0),
        dm_mu: float,
        dm_sigma: float = 0.3,
        dm_range: tuple[float, float],
        M_met: int = 200,
        M_loga: int = 200,
        pms_column: str | None = None,
        pms_max: float = 0.5,
        ms_weight: float = 1.0,
    ) -> IsochroneFitter:
        """Create and prepare an IsochroneFitter without computing the grid.

        Returns the fitter so you can call ``set_priors()``, ``build_grid()``,
        and ``fit()`` as separate steps in the notebook.

        Parameters
        ----------
        isochs_path : path
            Folder containing MIST ``.iso.cmd`` files.
        cluster : int, optional
            Cluster label (defaults to ``selected_cluster``).
        prob_threshold : float
            Minimum membership probability for CMD members.
        probability_column : str, default ``"probability_hdbscan"``
            Cual columna umbralizar. **Difiere a proposito** de los metodos de estructura
            (`center_determination`, `half_mass_radius`, ...), que usan ``"probability"`` =
            ``probabilities_ x probability_times`` y por tanto siempre <= esta. El ajuste de
            isocrona conserva ``probability_hdbscan`` porque el manuscrito de NGC 6383 esta
            enviado; ver el docstring de :mod:`erotica._membership`.

        Returns
        -------
        fitter : IsochroneFitter
            Ready for ``set_priors()`` → ``build_grid()`` → ``fit()``.

        Examples
        --------
        >>> # `loga_range`, `dm_mu` y `dm_range` son OBLIGATORIOS y son del CUMULO que estas
        >>> # ajustando, no de un default: los de abajo son de NGC 6383. Para otro cumulo, otros.
        >>> fitter = ca.prepare_isochrone_fitter(
        ...     isochs_path="./MIST/",
        ...     cluster=12,
        ...     loga_range=(6.0, 7.0),
        ...     dm_mu=10.2,
        ...     dm_range=(9.5, 10.7),
        ... )
        >>> fitter.set_priors({"dm_mu": 10.2, "dm_sigma": 0.3, "loga_range": (6.0, 7.0)})
        >>> fitter.build_grid(M_met=200, M_loga=200, grid_cache="./data/40/hgrid.npz")
        >>> idata = fitter.fit(draws=2000, tune=1000, chains=4)
        """

        cid = cluster if cluster is not None else self.selected_cluster
        if cid is None:
            raise ValueError("No cluster specified; call select_cluster() first or pass `cluster`.")
        cluster_data = self.data[self.data["cluster"] == cid]

        fitter = IsochroneFitter(
            isochs_path=isochs_path,
            loga_range=loga_range,
            Av_range=Av_range,
            dm_mu=dm_mu,
            dm_sigma=dm_sigma,
            dm_range=dm_range,
            M_met=M_met,
            M_loga=M_loga,
        )
        fitter.setup(
            cluster_data,
            prob_threshold=prob_threshold,
            probability_column=probability_column,
            precompute_grid=False,
            pms_column=pms_column,
            pms_max=pms_max,
            ms_weight=ms_weight,
        )
        return fitter

    def fit_isochrone(
        self,
        *,
        isochs_path: str | Path,
        cluster: int | None = None,
        prob_threshold: float = 0.6,
        probability_column: str = COLUMNA_ISOCRONA,
        loga_range: tuple[float, float],
        Av_range: tuple[float, float] = (0.0, 3.0),
        dm_mu: float,
        dm_sigma: float = 0.3,
        dm_range: tuple[float, float],
        M_met: int = 200,
        M_loga: int = 200,
        grid_cache: str | Path | None = None,
        draws: int = 2000,
        tune: int = 1000,
        chains: int | None = None,
        cores: int | None = None,
        target_accept: float = 0.8,
        nuts_sampler: str = "pymc",
        random_seed: int | None = None,
        init: str = "auto",
        initvals: dict | None = None,
        log_likelihood: bool = False,
        nuts_sampler_kwargs: dict | None = None,
    ) -> tuple[IsochroneFitter, Any]:
        """Fit an isochrone to the CMD of a cluster via Bayesian inference.

        Returns
        -------
        fitter : IsochroneFitter
            Configured fitter (holds grid, synthcl, extinction coefficients).
        idata : arviz.InferenceData
            Full posterior, including diagnostics.

        Examples
        --------
        >>> fitter, idata = ca.fit_isochrone(
        ...     isochs_path="./MIST/",
        ...     cluster=12,
        ...     loga_range=(6.0, 7.0),
        ...     dm_mu=10.3,
        ...     dm_sigma=0.2,
        ...     dm_range=(9.5, 10.7),
        ...     draws=2000,
        ...     tune=1000,
        ... )
        """
        from pathlib import Path as _Path

        cid = cluster if cluster is not None else self.selected_cluster
        if cid is None:
            raise ValueError("No cluster specified; call select_cluster() first or pass `cluster`.")
        cluster_data = self.data[self.data["cluster"] == cid]

        fitter = IsochroneFitter(
            isochs_path=isochs_path,
            loga_range=loga_range,
            Av_range=Av_range,
            dm_mu=dm_mu,
            dm_sigma=dm_sigma,
            dm_range=dm_range,
            M_met=M_met,
            M_loga=M_loga,
        )

        if grid_cache is not None and _Path(grid_cache).exists():
            print(f"Loading H_grid from cache: {grid_cache}")
            fitter.setup(
                cluster_data,
                prob_threshold=prob_threshold,
                probability_column=probability_column,
                precompute_grid=False,
            )
            fitter.load_grid(grid_cache)
        else:
            fitter.setup(
                cluster_data,
                prob_threshold=prob_threshold,
                probability_column=probability_column,
            )
            if grid_cache is not None:
                fitter.save_grid(grid_cache)
                print(f"H_grid saved to: {grid_cache}")

        idata = fitter.fit(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            nuts_sampler=nuts_sampler,
            random_seed=random_seed,
            init=init,
            initvals=initvals,
            log_likelihood=log_likelihood,
            nuts_sampler_kwargs=nuts_sampler_kwargs,
        )
        return fitter, idata


__all__ = ["ClusterAnalyzer"]
