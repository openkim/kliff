Running the model in LAMMPS or ASE
==================================

Dependencies
------------

Physics based KIM models can run from LAMMPS or ASE out of the box. You just need to install them as

.. code-block:: bash

    kim-api-collections-management install system <model name>

However, for running the ML models you need install few dependencies first, namely
- libtorch (C++ API for Pytorch)
- libtorchscatter and libtorchsparse

.. warning::

    Installing libtorch in same ``LD_LIBRARY_PATH`` as Pytorch can crash your pytorch installation (segfaults), which will mean that you will need to remove the libtorch from
    the environment path everytime you neet to train. Therefore it is recommended that you create **a new conda env, specifically for running the models**.


Let us create a fresh environment and install required dependencies

.. code-block:: shell

    conda create -n kliff_run
    conda activate kliff_run
    conda install lammps kimpy ase=3.22 python=3.9 unzip -c conda-forge

If you want to use the descriptor based models as well, then install ``libdescriptor``, and set its root path

.. code-block:: shell

    conda install libdescriptor -c ipcamit -c conda-forge
    export LIBDESCRIPTOR_ROOT=$CONDA_PREFIX


.. tip::

    If you are using GPU based systems, then it is advisable to use CMake version 3.18, with recommended CUDA version 11.7.
    Newer versions of CMake fail to detect CUDA environments correctly. You can install CMake as

.. code-block:: shell

    conda install cmake=3.18 -c conda-forge


Now, installing libtorch and other dependencies is simple, for most systems you can just download the binaries and use them directly.
If you are working in a Linux environment, OpenKIM provides an easy-to-use installation script that you can use as:

.. code-block:: shell

    wget https://openkim.org/files/MD_173118614730_000/install_dependencies.sh
    bash install_dependencies.sh

For new Apple MACs please use:

Following the installation, just source the produces `env.sh` file:

.. code-block:: shell

    source env.sh

and now you are ready to install the required model driver

.. code-block:: shell

    kim-api-collections-management install system TorchML__MD_173118614730_000


Following this you can install your models as same

.. code-block:: shell

    kim-api-collections-management install system SchNet1__MO_000000000000_000

TEST SW

Using your models
-----------------

Now that your model is installed, you can use it in LAMMPS and ASE,

ASE
^^^

RUN with SW first.

.. code-block:: python

    from ase.calculators.kim.kim import KIM
    from ase.build import bulk

    si = bulk("Si")
    model = KIM("SchNet1__MO_000000000000_000")
    si.calc = model
    print(si.get_potential_energy())
    print(si.get_forces())

LAMMPS
^^^^^^

.. code-block:: bash

    # Define KIM model and get Si diamond lattice parameter for this potential
    kim init         SchNet1__MO_000000000000_000 metal
    kim query        a0 get_lattice_constant_cubic crystal=["diamond"] species=["Si"] units=["angstrom"]
    # Setup diamond crystal
    boundary         p p p
    lattice          diamond ${a0}
    region           simbox block 0 1 0 1 0 1 units lattice
    create_box       1 simbox
    create_atoms     1 box
    mass             1 28.0855
    # Define atom type to species mapping
    kim interactions Si
    # Compute energy
    run 0


Common Errors
-------------

1. ``std::optional error``

During the installation of dependencies (from install script) you might get an error looking
like

.. code-block::

    ... ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:201:37: note: 'std::integral_constant<bool, false>::value' evaluates to false
        did you forgot #include <optional>
        Requires CXX17 support..

This indicates that for some reason you are installing a mismatched version of torch scatter
or torchsparse libraries. Either try to download the install script again, or try the copy of
latest install script as

.. code-block:: shell

    wget https://gist.githubusercontent.com/ipcamit/646573856b7f5735edd7048687d9655a/raw/d9e2cd51436b3f0a14ec5366f2335575801831fe/install_dependencies.sh

2. Python not found

.. code-block::

    CMake Error at CMakeLists.txt:28 (add_library):
    Target "torchscatter" links to target "Python3::Python" but the target was
    not found.  Perhaps a find_package() call is missing for an IMPORTED
    target, or an ALIAS target is missing?

This error usually means that you are using python > 3.9 with CMake == 3.18. CMake 3.18
has hardcoded python version <=3.9 string, making it yield an error. You have two options,

i. use CMAKE 3.28 and no CUDA/GPU
ii. use CMAKE 3.18 python <= 3.9

Another option for using python >3.9 with CMAKE 3.18 is to patch the the CMAKE python module as

.. code-block:: shell

    sed -i "25s/set(_\${_PYTHON_PREFIX}_VERSIONS /set(_\${_PYTHON_PREFIX}_VERSIONS $(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")') /" $CONDA_PREFIX/share/cmake-3.18/Modules/FindPython/Support.cmake

.. tip::

    This will only work on CMake 3.18, and only do it if you absolutely need the GPU support.
    It prepends your current python version to the Line 25 of CMake module.


Making changed permanent
-------------------------

.. warning::
    This can potentially messup your conda install. So only do tis if you understand the risks.

If you want to ensure that everytime you activate your conda environment it loads all the
dependencies on itself, without `source env.sh`, you can create activation hooks in conda
environment. Please make sure that you are in same folder as your `env.sh` file and run,

.. code-block:: shell

    conda activate kliff_run

    [[ -f env.sh ]] || { echo "Error: env.sh not found"; return; }

    # Create hooks directories
    mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

    env_prefix=$CONDA_PREFIX

    # Generate activation script
    echo "#!/bin/bash" > "$CONDA_PREFIX/etc/conda/activate.d/env_activate.sh"
    echo "export _OLD_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH" >> "$CONDA_PREFIX/etc/conda/activate.d/env_activate.sh"
    echo "export _OLD_PATH=\$PATH" >> "$CONDA_PREFIX/etc/conda/activate.d/env_activate.sh"
    cat env.sh >> "$CONDA_PREFIX/etc/conda/activate.d/env_activate.sh"

    conda deactivate
    # Generate deactivation script
    echo "#!/bin/bash" > "$env_prefix/etc/conda/deactivate.d/env_deactivate.sh"
    echo "export LD_LIBRARY_PATH=\$_OLD_LD_LIBRARY_PATH" >> "$env_prefix/etc/conda/deactivate.d/env_deactivate.sh"
    echo "export PATH=\$_OLD_PATH" >> "$env_prefix/etc/conda/deactivate.d/env_deactivate.sh"
    echo "unset _OLD_LD_LIBRARY_PATH" >> "$env_prefix/etc/conda/deactivate.d/env_deactivate.sh"
    echo "unset _OLD_PATH" >> "$env_prefix/etc/conda/deactivate.d/env_deactivate.sh"


