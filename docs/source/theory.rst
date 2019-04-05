.. _theory:

======
Theory
======

A parametric potential typically takes the form

.. math::
    \mathcal{V} = \mathcal{V}(\bm r_1,\dots,\bm r_{N_a}, Z_1,\dots,Z_{N_a}; \bm\theta)

where :math:`\bm r_1,\dots,\bm r_{N_a}` and :math:`Z_1,\dots,Z_{N_a}` are the
coordinates and species of a system of :math:`N_a` atoms, respectively, and
:math:`\bm\theta` denotes a set of fitting parameters.
For notational simplicity, in the following discussion, we assume that the atomic
species information is implicitly carried by the coordinates and thus we can exclude
:math:`Z` from the functional form, and use :math:`\bm R` to denote the
coordinates of all atoms in the configuration. Then we have

.. math::
    \mathcal{V} = \mathcal{V}(\bm R; \bm\theta).

A potential parameterization process is typically formulated as a weighted
least-squares minimization problem, where we adjust the potential parameters
:math:`\bm\theta` so as to reproduce a training set of reference data obtained from
experiments and/or first-principles computations. Mathematically, we hope to minimize a loss function

.. math::
    \mathcal{L(\bm\theta)} = \frac{1}{2} \sum_{i=1}^{N_p}
    w_i \|\bm p_i(\mathcal{V}(\bm R_i; \bm\theta)) - \bm q_i \| ^2

with respect to :math:`\bm\theta`, where :math:`\{\bm q_1,\dots, \bm q_{N_p}\}` is
a training set of :math:`N_p` reference data, :math:`\bm p_i` is the corresponding
prediction for :math:`\bm q_i` computed from the potential (as indicated by its
argument), :math:`\|\cdot\|` denote the :math:`L_2` norm, and :math:`w_i` is the
weight for the :math:`i`-th data point. We call

.. math::
    \bm u = \bm p(\mathcal{V}(\bm R; \bm\theta)) - \bm q

the residual function that characterizes the difference between the potential
predictions and the reference data for a set of properties.

Generally speaking, :math:`\bm q` can be a collection of any material properties
considered important for a given application, such as the cohesive energy,
equilibrium lattice constant, and elastic constants of a given crystal phase.
These materials properties can be obtained from experiments and/or
first-principles calculations.
However, nowadays, most of the potentials are trained using the `force-matching`
scheme, where the potential is trained to a large set of forces on atoms
(and/or energies, stresses) obtained by first-principles calculations for a
set of atomic configurations. This is extremely true for machine learning
potentials, where a large set of training data is necessary, and it seems impossible
to collect sufficient number of material properties for the training set.

The reference :math:`\bm q` and the prediction :math:`\bm p` are typically
represented as vectors such that
:math:`p[m]` is the :math:`m`-th reference property and :math:`p[m]` is the
corresponding :math:`m`-th prediction obtained from the potential.
Assuming we want to fit a potential to energy and forces, then :math:`\bm q`
is a vector of size :math:`1+3N_a`, in which :math:`N_a` is the number
of atoms in a configuration, with

.. math::
    q[0] &= E_\text{ref}\\
    q[1] &= f_\text{ref}^{0, x}, \quad
    q[2] = f_\text{ref}^{0, y}, \quad
    q[3] = f_\text{ref}^{0, z}, \\
    q[4] &= f_\text{ref}^{1, x}, \quad
    q[5] = f_\text{ref}^{1, y}, \quad
    q[6] = f_\text{ref}^{1, z}, \\
    \cdots \\
    q[3N_a-2] &= f_\text{ref}^{N_a-1, x}, \quad
    q[3N_a-1] = f_\text{ref}^{N_a-1, y}, \quad
    q[3N_a] = f_\text{ref}^{N_a-1, z}, \\

where :math:`E_\text{ref}` is the reference energy, and :math:`f_\text{ref}^{i, x}`,
:math:`f_\text{ref}^{i, y}`, and :math:`f_\text{ref}^{i, z}` denote the
:math:`x`-, :math:`y`-, and :math:`z`-component of reference force on atom
:math:`i`, respectively.
In other words, we put the energy as the 0th component of :math:`\bm q`, and
then put the force on the first atom as the 1st to 3rd components of  :math:`\bm q`,
the force on the second atom the next three components till the forces on all
atoms are placed in :math:`\bm q`.
In the same fashion, we can construct the prediction vector :math:`\bm p`, and
then to compute the residual vector.

.. note::
    We use boldface with subscript to denote a data point (e.g. :math:`\bm q_i`
    means the  :math:`i`-th data point in the training set), and use normal text
    with square bracket to denote the component of a data point (e.g. : :math:`q[m]`
    indicates the :math:`m`-th component of a general data point :math:`\bm q`.

If stress is used in the fitting, :math:`q[3N_a]` to :math:`q[3N_a+5]` will store
the reference Voigt stress
:math:`\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{yz}, \sigma_{xy}, \sigma_{xz}`,
and, of course, :math:`p[3N_a]` to :math:`p[3N_a+5]` are the corresponding
predictions computed from the potential.

The objective of the parameterization process is to find a set of parameters
:math:`\bm\theta` of potential that reproduce the reference data as well as
possible.

