# Release a new version

- Change the version by modifying `major`, `minor` and `patch` in `./update_version.py`
  and then `$ python update_version.py`

- Format code
   ```shell
   $ mamba update -c conda-forge black
   $ python format_sources.py
   ````

- Update `kliff/docs/source/changelog.rst`

- Update docs at `kliff/docs/source` as necessary

- Generate docs by running `$ make html` in the `kliff/docs` directory
  - remove `.git/hooks/pre-commit` so that it will not correct the generated
    doc files, otherwise ReadTheDoc will try to regenerate it and then fail
  - after commit it, then `$ pre-commit install` to get pre-commit back

- Commit and merge it to the `docs` branch. [ReadTheDocs](https://readthedocs.org)
  is set up to watch this branch and will automatically generate the docs.)

- Go to GitHub and release a new version by tagging it with `v<major><minor><patch>`.
  The GitHub actions at `kliff/.github/workflows/pythonpublish.yml` will then
  automatically publish it to PyPI
