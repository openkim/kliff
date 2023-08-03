# Contributing guide

## Code style

KLIFF adopts the [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings in Python code.
Other docs like this file is written in [MyST](https://myst-parser.readthedocs.io/en/latest/syntax/typography.html) markdown, which is an extension of the standard markdown. If you are familiar with the standard markdown, it should be easy to write in MyST.

KLIFF uses [isort](https://pycqa.github.io/isort/), [black](https://black.readthedocs.io/en/stable/) a set of other tools to format and statically check the correctness of the code.
These tools are already configured using `pre-commit`. To use it,

### Install pre-commit

```shell
pip install pre-commit
```

### Install pre-commit hooks for KLIFF

```shell
cd kliff
pre-commit install
```

### Run pre-commit checks

```shell
pre-commit run --all-files --show-diff-on-failure
```

This will run all the checks on all files. If there are warnings and errors reported, you can fix them and then commit the changes.

```{note}
After `pre-commit install` the checks will be run automatically before each commit. You can do `pre-commit uninstall` to disable the checks.
```

## Build the docs locally

To generate the docs (including the tutorials) locally on your machine, do

```shell
cd kliff/docs
make html
```

The generated docs will be at `kliff/docs/build/html/index.html`, and you can open it in a browser.

##Tutorials

If you have great tutorials, please write it in Jupyter notebook and place them in the `kliff/docs/tutorials` directory. Then update the `kliff/docs/tutorials.rst` file to include the new tutorials. After this, the tutorials will be automatically built and included in the docs.

In your Jupyter notebook, you can use MyST markdown to write the text.

```{warning}
We are in the process of migrating some of the docs from RestructuredText to MyST markdown. So you may see some of the docs are written in RestructuredText, and some links may be broken. We will fix them soon.
```
