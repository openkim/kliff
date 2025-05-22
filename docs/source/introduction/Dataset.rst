Practical Introduction to the Dataset Module
============================================

Newer KLIFF introduces lots more functionality towards dataset io while
maintaining backward compatibility. In this example we will go over the
dataset module and functionalities.

Dataset and Configuration
-------------------------

The dataset module contains two classes ``Dataset`` and
``Configuration``.

Configuration
~~~~~~~~~~~~~

``Configuration`` class contains the single unit of trainable data in a
dataset, which is

+------------+-----------------------------------------------+------------------------------+-------------------------------+
| Sr. no.    | Data                                          | Class Member Name            | Data type                     |
+============+===============================================+==============================+===============================+
| 1          | Coordinates of the atoms in the configuration | ``coords``                   | ``numpy.float64`` array       |
+------------+-----------------------------------------------+------------------------------+-------------------------------+
| 2          | Species                                       | ``species``                  | List of atomic symbols ``str``|
+------------+-----------------------------------------------+------------------------------+-------------------------------+
| 3          | “Global” energy of the configuration          | ``energy``                   | Python ``float`` (double-     |
|            |                                               |                              | precision)                    |
+------------+-----------------------------------------------+------------------------------+-------------------------------+
| 4          | Per-atom forces of the configuration          | ``forces``                   | ``numpy.float64`` array       |
|            |                                               |                              | (same shape as ``coords``)    |
+------------+-----------------------------------------------+------------------------------+-------------------------------+
| 5          | Periodic boundaries of the configuration      | ``PBC``                      | List of length 3 ``bool``     |
|            |                                               |                              | (periodicity in X, Y, Z)      |
+------------+-----------------------------------------------+------------------------------+-------------------------------+
| 6          | Cell vectors (row-wise, i.e. ``cell[0,:]`` is | ``cell``                     | 3 × 3 ``numpy.float64`` array |
|            | the first vector and ``cell[2,:]`` the last)  |                              |                               |
+------------+-----------------------------------------------+------------------------------+-------------------------------+
| 7          | Global stress on the configuration            | ``stress``                   | ``numpy.ndarray`` of shape    |
|            |                                               |                              | ``(6,)`` (Voigt notation)     |
+------------+-----------------------------------------------+------------------------------+-------------------------------+
| 8          | Weight to apply to this configuration during  | ``weight``                   | Instance of ``Weight`` class  |
|            | training                                      |                              |                               |
+------------+-----------------------------------------------+------------------------------+-------------------------------+
| 9          | Structural fingerprint of the configuration   | ``fingerprint``              | ``Any`` – typically a numpy   |
|            | (descriptors, graphs, etc.)                   |                              | array, torch tensor, or       |
|            |                                               |                              | :py:class:`~PyGGraph` object  |
+------------+-----------------------------------------------+------------------------------+-------------------------------+
| 10         | Per-configuration metadata key–value pairs    | ``metadata``                 | ``dict``                      |
+------------+-----------------------------------------------+------------------------------+-------------------------------+


.. warning::

    ASE Version Current `Configuration` method works with `ase` <= 3.22. So please pin to that version. Support for newer `ase` modules will be introduced next.


You can easily initialize the ``Configuration`` from ``ase.Atoms``

.. code-block:: python

    import numpy as np
    from ase.build import bulk
    
    from kliff.dataset import Configuration
    
    Si = bulk("Si")
    configuration = Configuration.from_ase_atoms(Si)
    print(configuration.coords)
    print(configuration.species)


.. parsed-literal::

    [[0.     0.     0.    ]
     [1.3575 1.3575 1.3575]]
    ['Si', 'Si']


There are other IO functions to directly initialize the Configuration
class, e.g.

1. ``Configuration.from_file`` : using extxyz file
2. ``Configuration.from_colabfit`` : using ColabFit exchange database

But it is best to use the ``Dataset`` to directly load these
configurations, as the ``Dataset`` object is more equipped to handle any
exceptions in reading these files.

Direct initialization
~~~~~~~~~~~~~~~~~~~~~

For conversion to newer or unsupported data formats, you can directly
initialize the configuration object as

.. code-block:: python

    cell = np.eye(3)  # 3x3 identity matrix
    species = ["Al", "Al", "Al", "Al"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
    ])
    pbc = [True, True, True]
    
    config = Configuration(
        cell=cell,
        species=species,
        coords=coords,
        PBC=pbc,
        energy=-3.5,
        forces=np.random.randn(4, 3),  # random forces as an example
        stress=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Voigt notation
    )
    
    # Let's print some info:
    print("Number of atoms:", config.get_num_atoms())
    print("Species:", config.species)
    print("Energy:", config.energy)
    print("Forces:\n", config.forces)



.. parsed-literal::

    Number of atoms: 4
    Species: ['Al', 'Al', 'Al', 'Al']
    Energy: -3.5
    Forces:
     [[ 1.36756812 -1.39906188 -0.25229913]
     [-1.68647155  0.01372661 -0.30166477]
     [ 0.9050956  -0.08650277  0.28608345]
     [ 1.43834871  1.40225919 -0.14530453]]


Exporting the configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can convert configuration object back to {class}\ ``~ase.Atoms``
object using ``Configuration.to_ase_atoms``, or to extxyz file using
``Configuration.to_file``. For more details, please refer to the API
docs.

.. code-block:: python

    ase_atoms = configuration.to_ase_atoms()
    print(np.allclose(ase_atoms.get_positions(), configuration.coords))
    
    configuration.to_file("config1.extxyz")
    print("\nSaved extxyz header: ")
    print("="*80)


.. code-block:: bash

    head -2 config1.extxyz

.. tip::

    Commands with ``!`` in front runs in the shell in a Jupyter notebook. So please run
    them in shell if you are running these tutorials interactively


.. parsed-literal::

    True
    
    Saved extxyz header: 
    ================================================================================
    2
    Lattice="0 2.715 2.715 2.715 0 2.715 2.715 2.715 0" PBC="1 1 1" Properties=species:S:1:pos:R:3


Exception handling for ``Configuration``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If any absent property is accessed, you get ``ConfigurationError``
exception. User should handle these exceptions as they see fit.

.. code-block:: python

    configuration.forces # raises exception


::


    ---------------------------------------------------------------------------

    ConfigurationError                        Traceback (most recent call last)

    Cell In [4], line 1
    ----> 1 configuration.forces


    File ~/Projects/COLABFIT/kliff/kliff/kliff/dataset/dataset.py:378, in Configuration.forces(self)
        374 """
        375 Return a `Nx3` matrix of the forces on each atoms.
        376 """
        377 if self._forces is None:
    --> 378     raise ConfigurationError("Configuration does not contain forces.")
        379 return self._forces


    ConfigurationError: Configuration does not contain forces.


.. warning::

   `Configuration` does not store data with any notion of units, so ensuring the units of the io data is a user delegated responsibility.

Dataset
-------

Like mentioned earlier, ``Dataset`` is mostly a collection of
``Configurations``, with member functions to read and write those
configurations. In simplest terms the ``Dataset`` object works as a list
of ``Configurations``.

Initializing the ``Dataset``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can initialize the ``Dataset`` object using myraid of storage
options, which include:

1. List of ASE Atoms objects (with keyword ``ase_atoms_list`` eplicitly specified)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from kliff.dataset import Dataset
    
    configs = [bulk("Si"), bulk("Al"), bulk("Al", cubic=True)]
    ds = Dataset.from_ase(ase_atoms_list=configs)
    print(len(ds))


.. parsed-literal::

    2025-04-16 14:00:11.204 | INFO     | kliff.dataset.dataset:_read_from_ase:959 - 3 configurations loaded using ASE.
    2025-04-16 14:00:11.205 | INFO     | kliff.dataset.dataset:add_weights:1128 - No explicit weights provided.

    3


2. ``extzyz`` file (all configurations in single extxyz file, read using ``ase.io``, default behaviour)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us dowload a extyz dataset from web (in this case we are downloading
`Graphene dataset <https://doi.org/10.1038/s41467-023-44525-z>`__ in
extxyz format from Colabfit Exchange.

.. code:: bash

    # Download the dataset, and print header
    !wget https://materials.colabfit.org/dataset-xyz/DS_jasbxoigo7r4_0.tar.gz
    !tar -xvf DS_jasbxoigo7r4_0.tar.gz
    !xz -d DS_jasbxoigo7r4_0_0.xyz.xz
    !head -2 DS_jasbxoigo7r4_0_0.xyz


.. parsed-literal::

    --2025-04-16 14:00:11--  https://materials.colabfit.org/dataset-xyz/DS_jasbxoigo7r4_0.tar.gz
    Resolving materials.colabfit.org (materials.colabfit.org)... 216.165.12.42
    Connecting to materials.colabfit.org (materials.colabfit.org)|216.165.12.42|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 36567 (36K) [application/x-tar]
    Saving to: ‘DS_jasbxoigo7r4_0.tar.gz’
    
    DS_jasbxoigo7r4_0.t 100%[===================>]  35.71K  --.-KB/s    in 0.06s   
    
    2025-04-16 14:00:11 (600 KB/s) - ‘DS_jasbxoigo7r4_0.tar.gz’ saved [36567/36567]
    
    ./
    ./DS_jasbxoigo7r4_0_0.xyz.xz
    48
    Lattice="7.53 0.0 0.0 0.0 8.694891 0.0 0.0 0.0 6.91756" Properties=species:S:1:pos:R:3:forces:R:3 po_id=PO_1073537155164130421524433 co_id=CO_1056372038821617091165957 energy=-468.61686026192723 stress="-0.05233445077383756 0.003984624736573388 3.332094089548831e-06 0.003984624736573388 -0.03689214199484896 -6.99536080196756e-06 3.332094089548831e-06 -6.99536080196756e-06 -0.004744008663708218" pbc="T T T"


The things to note down in the header of the xyz file are the following,
i. ``Properties=species:S:1:pos:R:3:forces:R:3``, and ii.
``energy=-468.61686026192723``, as you might need to supply these energy
and forces keys (``forces`` and ``energy`` in above example) explicitly
to the function to ensure that properties are correctly mapped in KLIFF
configuration.

.. code-block:: python

    from kliff.utils import get_n_configs_in_xyz # how many configs in xyz file 
    # Read the dataset from DS_jasbxoigo7r4_0_0.xyz
    ds = Dataset.from_ase("./DS_jasbxoigo7r4_0_0.xyz", energy_key="energy", forces_key="forces")
    
    assert len(ds) == get_n_configs_in_xyz("./DS_jasbxoigo7r4_0_0.xyz")


.. parsed-literal::

    2025-04-16 14:00:13.031 | INFO     | kliff.dataset.dataset:_read_from_ase:959 - 41 configurations loaded using ASE.
    2025-04-16 14:00:13.032 | INFO     | kliff.dataset.dataset:add_weights:1128 - No explicit weights provided.


After loading the dataset you can use it as any other list, with simple
indices, slices, or list of numbers.

.. tip::

    Please note that slices and lists of config returns a new dataset object with
    desired configuration (as opposed to python list).

.. code-block:: python

    # access individual configs
    print(ds[1], ds[-1])
    
    # access slices
    print(len(ds[2:5]))
    
    # access using list of configs
    print(len(ds[1,3,5]))


.. parsed-literal::

    <kliff.dataset.dataset.Configuration object at 0x7f2d4757b970> <kliff.dataset.dataset.Configuration object at 0x7f2d4758eee0>
    3
    3


3. List of extxyz files (with one configuration per file)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dataset module can also be initialized using a list of xyz files, with
one configuration per file. Example below demonstrate on how to load a
toy dataset with 4 configurations.

.. code:: bash

    !wget https://raw.githubusercontent.com/openkim/kliff/main/examples/Si_training_set_4_configs.tar.gz
    !tar -xvf Si_training_set_4_configs.tar.gz


.. parsed-literal::

    --2025-04-16 14:00:13--  https://raw.githubusercontent.com/openkim/kliff/main/examples/Si_training_set_4_configs.tar.gz
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 2606:50c0:8000::154, 2606:50c0:8003::154, 2606:50c0:8002::154, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|2606:50c0:8000::154|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 7691 (7.5K) [application/octet-stream]
    Saving to: ‘Si_training_set_4_configs.tar.gz’
    
    Si_training_set_4_c 100%[===================>]   7.51K  --.-KB/s    in 0s      
    
    2025-04-16 14:00:13 (21.0 MB/s) - ‘Si_training_set_4_configs.tar.gz’ saved [7691/7691]
    
    Si_training_set_4_configs/
    Si_training_set_4_configs/Si_alat5.431_scale0.005_perturb1.xyz
    Si_training_set_4_configs/Si_alat5.409_scale0.005_perturb1.xyz
    Si_training_set_4_configs/Si_alat5.442_scale0.005_perturb1.xyz
    Si_training_set_4_configs/Si_alat5.420_scale0.005_perturb1.xyz


.. code-block:: python

    ds = Dataset.from_path("./Si_training_set_4_configs") # 4 configs in ./Si_training_set_4_configs
    assert len(ds) == 4


.. parsed-literal::

    2025-04-16 14:00:14.036 | INFO     | kliff.dataset.dataset:add_weights:1128 - No explicit weights provided.


4. From a ColabFit Exchange database instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also stream data from Colabfit Exchange as

.. code-block:: python

   ds = Dataset.from_colabfit("my_colabfit_database", "DS_xxxxxxxxxxxx_0", colabfit_uri = "mongodb://localhost:27017")

.. warning::

   The Colabfit interface is under heavy development so please check back for any changes till this warning is not removed

Exporting the dataset
~~~~~~~~~~~~~~~~~~~~~

You can export the dataset to different formats using ``to_<form>`` methods. Here, ``<form>``
can be ``ase``, ``path``, and ``colabfit``. For interactive inter-compatibility you can
also export the dataset to list of ``ase.Atoms`` objects using ``Dataset.to_ase_list`` method.


Custom Dataset Class
--------------------

For unsupported io formats, such as VASP, Siesta outfiles etc, you can
extend the ``Dataset`` class manually using the default
``Configuration.__init__`` method for populating the configurations. You
will need to store the list of loaded configurations in the
``Dataset.config`` member variable

.. code-block:: python

   class CustomDataset(Dataset):
       @classmethod
       def from_custom(files_path):
           self.config = []
           ... # get data from the file
           self.append(Configuration(cell=cell,
                                     species=species,
                                     coords=coords,
                                     PBC=pbc,
                                     energy=energy,
                                     forces=forces))

Weights
=======

KLIFF dataset configurations can have fine grained weights for training,
as provided by the :py:class:`~kliff.dataset.weight.Weight`.

