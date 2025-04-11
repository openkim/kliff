.. _doc.models:

Models
======

Model Class Documentation
=========================


The ``Model`` class defines a framework for representing parameterized models and dense
neural networks, especially for use in optimization workflows. This is intended to be
subclassed with user-defined behavior. Several methods must be implemented by any subclass.
The class provides a consistent interface for managing model parameters, for KIM-API
portable models.

.. note::

    For more complicated ML models, like NequIP, it is best to directly train the model
    without using the ``Model`` class. However, ``Model`` affords a better interface for
    KLIFF utilities, such as uq. In future releases of KLIFF, a more tightly integrated
    interface for GNNs will be provided for the ``Model`` class.

Constructor
-----------

.. py:method:: __init__(model_name=None)

    Initializes the model instance. You can optionally give your model a name.
    On initialization, it tries to set up parameters, supported species, and influence
    distance by calling methods you must define.

Required Methods (To Be Implemented in Subclass)
------------------------------------------------

.. py:method:: init_model_params()

    You must define this method to initialize and return a dictionary of model parameters
    (usually custom ``Parameter`` objects).

.. py:method:: init_influence_distance()

    Define this to return the interaction or influence distance of your model (a float).
    For graph neural networks, this is typically the cutoff distance times the number of
    convolutions.

.. py:method:: init_supported_species()

    Define this to return a dictionary of atomic species and its corresponding index.
    For example, ``{"H": 0, "O": 1}`` for a water model. This is used to determine the
    mapping between atomic species and their indices in simulators.

.. py:method:: get_compute_argument_class()

    Specific to KIM-API models. This should return the ComputeArguments.

.. py:method:: write_kim_model(path=None)

    Implement this if your model can be exported in KIM-API format.

Model Info Methods
------------------

.. py:method:: get_influence_distance()

    Returns the model's influence distance.

.. py:method:: get_supported_species()

    Returns a dictionary of supported atomic species.

.. py:method:: get_model_params()

    Returns the dictionary of parameters used in the model.

.. py:method:: echo_model_params(filename=sys.stdout)

    Prints or writes the current parameter values (both raw and transformed).

Parameter Configuration
-----------------------

.. py:method:: set_params_mutable(list_of_params)

    Marks specific parameters as optimizable (mutable). You must pass a list of parameter names.

.. py:method:: set_opt_params(**kwargs)

    Sets multiple parameters at once using keyword arguments. Usually,
    this will call ``set_one_opt_param()`` for each parameter.

.. py:method:: set_one_opt_param(name, settings)

    Allows fine-grained control of a single parameter’s value and bounds for optimization.

.. py:method:: echo_opt_params(filename=sys.stdout)

    Displays the values of parameters that are marked as optimizable.

.. py:method:: get_num_opt_params()

    Returns the number of total optimizable (mutable) parameter values (size of opt array).

.. py:method:: get_opt_params()

    Returns all optimizable values as a single NumPy array—useful for passing to an optimizer.

.. py:method:: update_model_params(params)

    Updates the model with a new set of parameter values (in the same shape as `get_opt_params()`).

.. py:method:: get_opt_param_name_value_and_indices(index)

    Given a global index, returns the parameter name, its value, and its index within the model. Helpful in debugging or tracking.

.. py:method:: get_formatted_param_bounds()

    Returns a tuple of bounds for all optimizable parameters—formatted for `scipy.optimize`.

.. py:method:: opt_params_has_bounds()

    Returns True if any of the optimizable parameters have bounds defined.

Saving and Loading
------------------

.. py:method:: save(filename="trained_model.yaml")

    Saves the mutable parameters of the model to disk as a YAML file.

.. py:method:: load(filename="trained_model.yaml")

    Loads the model's parameters from a YAML file (previously saved via `save()`).

Parameter Utilities
-------------------

.. py:method:: named_parameters()

    Returns a dictionary of all parameters currently marked as mutable/optimizable.

.. py:method:: parameters()

    Returns a list of mutable parameters for optimization (akin to ``torch.nn.Module.parameters()``).



KIM models
==========

Neural network models
=====================
