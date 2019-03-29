.. _tut_save_load_model:

=====================
Save and load a model
=====================

Once you've trained a model, you can save it disk and load it back later for the
purpose of retraining, evaluation, etc.


Save a model
============
The ``save()`` method of a model can be used to save it. Suppose you've trained
the Stillinger-Weber (SW) potential discussed in :ref:`tut_kim_sw`, you can save the
model by:

.. code-block:: python

    path = './kliff_model.pkl'
    model.save(path)

which creates a pickled file named ``kliff_model.pkl`` in the current working
directory. All the information related to the model are saved to the file,
including the final values of the parameters, the constraints on the parameters
(such as the bounds on parameters set via :meth:`~kliff.models.Model.set_one_fitting_param`
or :meth:`~kliff.models.Model.set_fitting_params`), and others.


Load a model
============

A model can be loaded using ``load()`` after the instantiation. For the same SW
potential discussed in :ref:`tut_kim_sw`, it can be loaded by:

.. code-block:: python

     model = KIM(model_name='Three_Body_Stillinger_Weber_Si__MO_405512056662_004')
     path = './kliff_model.pkl'
     model.load(path)

If you want to retrain the loaded model, you can attach it to a calculator and then
proceed as what discussed in :ref:`tut_kim_sw` and :ref:`tut_nn`.

