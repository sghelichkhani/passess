# passess documentation

The site is built with [Quarto](https://quarto.org) and
[quartodoc](https://machow.github.io/quartodoc/). Example notebooks are
maintained as [jupytext](https://jupytext.readthedocs.io/) `py:percent`
scripts under `../examples/` and converted to `.ipynb` at build time.

## One-time setup

1. Install Quarto (>=1.5). See https://quarto.org/docs/get-started/.
2. Install the Python docs dependencies into your working environment:

   ```bash
   pip install -e '.[docs]'
   ```

3. Make sure the Python kernel named `python3` is registered. If not:

   ```bash
   python -m ipykernel install --user --name python3
   ```

## Build

From the repository root:

```bash
docs/build.sh           # one-shot render into docs/_site
docs/build.sh preview   # live-reloading preview on http://localhost:4200
```

The script:

1. Converts every `examples/*.py` into a matching `docs/examples/*.ipynb`.
2. Runs `quartodoc build` to regenerate the API reference under
   `docs/reference/`.
3. Calls `quarto render` (or `quarto preview`).

`docs/_site/`, `docs/.quarto/`, `docs/reference/` (except the hand-written
`index.qmd`) and `docs/examples/*.ipynb` are all regenerated and are
gitignored.

## Deploy

The `docs.yml` GitHub Actions workflow rebuilds the site on every push to
`main` and publishes it to the `gh-pages` branch. To deploy manually:

```bash
cd docs
quarto publish gh-pages
```
