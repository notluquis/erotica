# Contributing

Bug reports, questions, and pull requests are welcome.

- **Bugs** → [GitHub issues](https://github.com/notluquis/erotica/issues)
- **Usage / methodology questions** → [GitHub discussions](https://github.com/notluquis/erotica/discussions)

## Development setup

```bash
git clone https://github.com/notluquis/erotica.git
cd erotica
pip install -e ".[dev]"
pytest -q
```

Branch off `dev`, add a test that fails without your change, run `black`/`isort`
(line length 100) and the test suite, then open a pull request against `dev`.

## Scientific conventions

EROTICA is a research package, so contributions are judged on scientific correctness as
much as on code quality:

- **Don't silently change a published default** — the NGC 6383 results under `data/test/`
  are a reproducibility artifact.
- **Options with disclosure** — a method that fails on some cluster morphology ships as an
  option with its failure mode documented, not as the default. See the
  [membership guide](guides/membership.md) for the pattern.
- **State what is estimated vs assumed** — priors, fixed parameters, and the model grid an
  age is conditioned on belong in the docstring.
- **Uncertainties are not decoration** — prefer a distribution or interval over a bare
  point estimate where the method supports it.

The full version, including the code of conduct, is in
[`CONTRIBUTING.md`](https://github.com/notluquis/erotica/blob/dev/CONTRIBUTING.md).
