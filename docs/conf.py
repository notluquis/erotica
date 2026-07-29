# docs/conf.py — EROTICA documentation build
import os
from importlib.metadata import version as _pkg_version

# -- Project information -----------------------------------------------------
project = "EROTICA"
author = "EROTICA developers"
copyright = "2026, EROTICA developers"
try:
    release = _pkg_version("erotica")  # distribution name in pyproject
except Exception:
    release = "0.0.1"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "numpydoc",          # NumPy-style docstrings — do NOT also enable napoleon
    "myst_parser",       # Markdown narrative pages + the design notes
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
# CLAUDE.md files are per-directory agent guidance, not documentation pages.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/CLAUDE.md"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]

# -- autodoc / autosummary ---------------------------------------------------
autosummary_generate = True          # generate stub pages from docstrings
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_typehints = "none"           # types come from numpy docstrings (no dup)
autoclass_content = "class"
autodoc_mock_imports = [             # heavy/optional deps so RTD builds without them
    "hdbscan", "optuna", "pymc", "arviz", "blackjax", "numpyro", "jax",
    "asteca", "gaiaunlimited", "fast_histogram", "dill", "adjusttext",
    "gaiadr3_zeropoint",
]

# -- numpydoc ----------------------------------------------------------------
numpydoc_show_class_members = False   # class template lists members; avoid dup
numpydoc_class_members_toctree = False

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "EROTICA"
html_theme_options = {
    "github_url": "https://github.com/notluquis/erotica",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "use_edit_page_button": True,
}
# Edit-on-GitHub button + canonical URLs (Read the Docs sets the env var on build).
html_context = {
    "github_user": "notluquis",
    "github_repo": "erotica",
    "github_version": "dev",
    "doc_path": "docs",
}
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
