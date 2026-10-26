"""Save a paper figure: the PDF that goes to the journal plus a PNG preview to look at."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure


def save_figure(
    fig: Figure, path: str | Path, *, preview_dpi: int = 300, **savefig_kwargs
) -> tuple[Path, Path]:
    """Write ``path`` (the publication artefact, normally a PDF) and ``<stem>_preview.png`` next to it.

    A one-column A&A figure is 9 cm wide at its true size, so a PDF viewer opens it too small to
    inspect. The preview is the same figure rasterised at ``preview_dpi`` (300 by default): open the
    preview to look at a figure, and send the PDF to the journal. Both are written at the figure's
    current size, which should be its printed size (see :func:`erotica.viz.figsize`).

    Extra keyword arguments go to both ``savefig`` calls (``dpi`` excepted for the preview).
    Returns ``(artefact_path, preview_path)``.
    """
    path = Path(path)
    preview = path.with_name(f"{path.stem}_preview.png")
    fig.savefig(path, **savefig_kwargs)
    kwargs = {k: v for k, v in savefig_kwargs.items() if k not in ("dpi", "format")}
    fig.savefig(preview, dpi=preview_dpi, format="png", **kwargs)
    return path, preview
