1. Run
   $ generate_apidoc.py
   to generate `tmp_apidoc`. Compare with these in `source/apidoc`, and move
   `tmp_apidoc` to `source_apidoc`. If everything looks good.

2. To generate docs:
   # generating sphinx-gallery examples (pop `source/auto_examples` directory)
   $ make html
   or
   # not generating sphinx-gallery examples (much faster)
   $ make html-notutorial
