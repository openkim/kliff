.. module:: doc.calculators

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

As a result, KLIFF provide several groups of functions to facilitate these
operations.

- `Initialization`:
  :meth:`~kliff.calculator.Calculator.create` and
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
