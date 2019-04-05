.. _doc.calculators:

===========
Calculators
===========

A calculator is the central agent that exchanges information between a model and
the minimizer.

- It uses the computation methods provided by a model to calculate the energy,
  forces, etc. and pass these the properties, together with the corresponding
  reference data, to :class:`~kliff.loss.Loss` to construct a cost function to be
  minimized by the optimizer.
- It also inquires the model to get parameters that are going to be optimized, and
  provide these parameters to the optimizer, which will be used as the initial values
  by the optimizer to carry out the optimization.
- In the reverse direction, at each optimization step, the calculator grabs the new
  parameters from the optimizer and update the model parameters with the new ones.
  So, in the next minimization step, the cost function will be calculated using the
  new parameters.


A calculator for the physics-motivated potential can be created by:

.. code-block:: python

    from kliff.calculator import Calculator

    model = ...  # create a model
    configs = ...  # get a list of configurations
    calc = Calculator(model)
    calc.create(configs, use_energy=True, use_forces=True, use_stress=False)

It creates a calculator for a ``model`` (discussed in :ref:`doc.models`), and
``configs`` is a list of :class:`~kliff.dataset.Configuration` (discussed in
:ref:`doc.dataset`), for which the calculator is going to make predictions.
``use_energy``, ``use_forces``, and ``use_stress`` inform the calculator whether
`energy`, `forces`, and `stress` will be requested from the calculator.
If the potential is to be trained on `energy` only, it would be better to set
``use_forces`` and ``use_stress`` to ``False``, which turns off the calculations for
``forces`` and ``stress`` and thus can speed up the fitting process.


Other methods of the calculator include:

- `Initialization`:
  :meth:`~kliff.calculator.Calculator.get_compute_arguments`.
- `Property calculation using a model`:
  :meth:`~kliff.calculator.Calculator.compute`,
  :meth:`~kliff.calculator.Calculator.get_compute_arguments`,
  :meth:`~kliff.calculator.Calculator.compute`,
  :meth:`~kliff.calculator.Calculator.get_energy`,
  :meth:`~kliff.calculator.Calculator.get_forces`,
  :meth:`~kliff.calculator.Calculator.get_stress`,
  :meth:`~kliff.calculator.Calculator.get_prediction`,
  :meth:`~kliff.calculator.Calculator.get_reference`.
- `Optimizing parameters`:
  :meth:`~kliff.calculator.Calculator.get_opt_params`,
  :meth:`~kliff.calculator.Calculator.get_opt_params_bounds`,
  :meth:`~kliff.calculator.Calculator.update_opt_params`.

.. seealso::
    See :class:`kliff.calculator.Calculator` for a complete list of the member
    functions and
    their docs.
