# Release a new version

- Change the version by modifying `major`, `minor` and `patch` in `./update_version.py`
  and then `$ python update_version.py`

- Format code

  ```shell
  $ mamba update -c conda-forge black
  $ python format_sources.py
  ```

- Update `CHANGELOG.md`

- Update docs at `kliff/docs/source` as necessary

- Generate docs by running `$ make html` in the `kliff/docs` directory

  - Note, check the generated tutorial notebooks -- sometimes sphinx-galley will
    not correctly capture the stdout and embed it in the file. In this case,
    delete hte .md5 file and return `make html`. (Also, do it one be one.)
    For example,
    ```shell
    $ rm source/auto_examples/example_kim_SW_Si.py.md5
    $ make html
    ```

- Commit and merge it to the `docs` branch. [ReadTheDocs](https://readthedocs.org)
  is set up to watch this branch and will automatically generate the docs.)

- Go to GitHub and release a new version by tagging it with `v<major><minor><patch>`.
  The GitHub actions at `kliff/.github/workflows/pythonpublish.yml` will then
  automatically publish it to PyPI
