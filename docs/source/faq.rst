.. _faq:

===========================
Frequently Asked Questions
===========================


.. _kim_fails:

I am using a KIM model, but it fails. What should I do?
=======================================================
- Check you have the model installed. You can use ``$ kim-api-collections-management
  list`` to see what KIM models are installed.
- Check ``kim.log`` to see what the error is. Note that ``kim.log`` stores all the
  log info chronologically, so you may want to delete it and run you fitting code
  to get a fresh one.
- Make sure that parameters like ``cutoff``, ``rhocutoff`` is not used as fitting
  parameters. See ":ref:`simulator_neighbor_error` for more".


.. _simulator_neighbor_error:

What does "error * * Simulator supplied GetNeighborList() routine returned error" in ``kim.log`` mean?
======================================================================================================
Probably you use parameters related to cutoff distance (e.g. ``cutoff`` nad
``rhocutoff``) as fitting parameters. KLIFF build neighbor list only once at the
beginning, and reuse it during the optimization process. If the cutoff changes,
the neighbor list could be invalid any more. Typically, in the training of
potentials, we treat cutoffs as predefined hyperparameters and do not optimize
them. So simply remove them from your fitting parameters.
