## Summary

Include a summary of major changes in bullet points:

* Feature 1
* Fix 1

## Additional dependencies introduced (if any)

* List all new dependencies needed and justify why. While adding dependencies that bring
  significantly useful functionality is perfectly fine, adding ones that add trivial
  functionality, e.g., to use one single easily implementable function, is frowned upon.
  Provide a justification why that dependency is needed. Especially frowned upon are
  circular dependencies.

## TODO (if any)

If this is a work-in-progress, write something about what else needs to be done.

* Feature 1 supports A, but not B.

## Checklist

Before a pull request can be merged, the following items must be checked:

* [ ] Make sure your code is properly formatted. [isort](https://pycqa.github.io/isort/) and [black](https://black.readthedocs.io/en/stable/getting_started.html) are used for this purpose. The simplest way is to use [pre-commit](https://pre-commit.com). See instructions [here](https://github.com/openkim/kliff/blob/main/docs/source/contributing_guide.md#code-style).
* [ ] Doc strings have been added in the [Google docstring format](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html) on your code.
* [ ] Type annotations are **highly** encouraged. Run [mypy](http://mypy-lang.org) to
  type check your code.
* [ ] Tests have been added for any new functionality or bug fixes.
* [ ] All linting and tests pass.

Note that the CI system will run all the above checks. But it will be much more
efficient if you already fix most errors prior to submitting the PR.
 