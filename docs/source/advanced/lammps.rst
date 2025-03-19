.. _lammps:

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


.. attention::

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

.. tab-set::
    :sync-group: category

    .. tab-item:: Linux
        :sync: linux

        .. code-block:: shell

            wget https://openkim.org/files/MD_173118614730_001/install_dependencies.sh
            bash install_dependencies.sh

    .. tab-item:: MacOS(Arm64)
        :sync: macos

        .. code-block:: shell

            wget https://gist.githubusercontent.com/ipcamit/25b9de5ab823fe00f8eaae89ef8c60d4/raw/1c2cf62ef3e9f2108e5227feb9cb50d203422a35/install_macos_dependencies.sh
            bash install_macos_dependencies.sh


Following the installation, just source the produces `env.sh` file:

.. tab-set::
    :sync-group: category

    .. tab-item:: Linux
        :sync: linux

        .. code-block:: shell

            source env.sh

    .. tab-item:: MacOS(Arm64)
        :sync: macos

        .. code-block:: shell

            source env.sh

            # For runtime
            cp $TorchSparse_ROOT/lib/libtorchsparse.dylib  $CONDA_PREFIX/lib/
            cp $TorchScatter_ROOT/lib/libtorchscatter.dylib  $CONDA_PREFIX/lib/
            cp $TORCH_ROOT/lib/*  $CONDA_PREFIX/lib/

and now you are ready to install the required model driver

.. note::
    KIM-API ``system`` install location on MacOS is a hit-or-miss, so it is recommended to use ``user`` install location for MacOS.
    Also due to older ``libtorch`` versions not being available for MacOS, you need to install the TorchML driver manually.
    This is mainly because newer versions of ``libtorch`` specifically need C++17 support.

.. tab-set::
    :sync-group: category

    .. tab-item:: Linux
        :sync: linux

        .. code-block:: shell

            kim-api-collections-management install user TorchML__MD_173118614730_000

    .. tab-item:: MacOS(Arm64)
        :sync: macos

        .. code-block:: shell

            wget https://openkim.org/download/TorchML__MD_173118614730_000.txz
            tar -xvf TorchML__MD_173118614730_000.txz
            sed -i '' 's/libdescriptor.so/libdescriptor.dylib/' TorchML__MD_173118614730_000/CMakeLists.txt
            sed -i '' 's/PROPERTY CXX_STANDARD 14/PROPERTY CXX_STANDARD 17/' TorchML__MD_173118614730_000/MLModel/CMakeLists.txt
            kim-api-collections-management install user TorchML__MD_173118614730_000


Testing the installation
------------------------

Let us install a simple Stillinger-Weber potential for Silicon and test it

.. code-block:: shell

    kim-api-collections-management install user  SW_StillingerWeber_1985_Si__MO_405512056662_006

.. code-block:: python

    from ase.calculators.kim.kim import KIM
    from ase.build import bulk

    si = bulk("Si")
    model = KIM("SW_StillingerWeber_1985_Si__MO_405512056662_006")
    si.calc = model
    print(si.get_potential_energy())
    print(si.get_forces())


Using your models
-----------------

You can install you models now as any other KIM model,

.. code-block:: shell

    kim-api-collections-management install user SchNet1__MO_111111111111_000

Now that your model is installed, you can use it in LAMMPS and ASE,

ASE
^^^

RUN with SW first.

.. code-block:: python

    from ase.calculators.kim.kim import KIM
    from ase.build import bulk

    si = bulk("Si")
    model = KIM("SchNet1__MO_111111111111_000")
    si.calc = model
    print(si.get_potential_energy())
    print(si.get_forces())

LAMMPS
^^^^^^

Save the following script as `in.lammps`,

.. code-block:: bash

    # Define KIM model and get Si diamond lattice parameter for this potential
    kim init         SchNet1__MO_111111111111_000 metal
    # Setup diamond crystal
    boundary         p p p
    lattice          diamond 5.44
    region           simbox block 0 1 0 1 0 1 units lattice
    create_box       1 simbox
    create_atoms     1 box
    mass             1 28.0855
    # Define atom type to species mapping
    kim interactions Si
    # Compute energy
    run 0


You can now run the LAMMPS script as

.. code-block:: shell

    lmp_serial -i in.lammps


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

i. use latest CMAKE and no CUDA/GPU
ii. use CMAKE 3.18 python <= 3.9

Another option for using python >3.9 with CMAKE 3.18 is to patch the the CMAKE python module as

.. code-block:: shell

    sed -i "25s/set(_\${_PYTHON_PREFIX}_VERSIONS /set(_\${_PYTHON_PREFIX}_VERSIONS $(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")') /" $CONDA_PREFIX/share/cmake-3.18/Modules/FindPython/Support.cmake

.. tip::

    This will only work on CMake 3.18, and only do it if you absolutely need the GPU support.
    It prepends your current python version to the Line 25 of CMake module.


3. ``dyld[19429]: symbol not found in flat namespace '_error_top'`` in lammps

LAMMPS is not properly installed. You can try to reinstall LAMMPS,

.. code-block:: shell

    conda remove lammps
    conda install lammps -c conda-forge

4. GLIBCXX_3.X.X not found

.. code-block::

    version `GLIBCXX_3.4.29' not found (required by /opt/mambaforge/mambaforge/envs/kliff_run/lib/libkim-api.so.2)

This error usually means that you are using a different version of GCC than the one used to compile the KIM-API.
Just set the following environment variable before running the code,

.. code-block:: shell

    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

5. ``lmp_serial command not found``

Sometimes in some conda environments, the LAMMPS binary is called ``lmp``, so try

.. code-block:: shell

    lmp -in in.lammps

6. Missing KIM model during ASE/LAMMPS runs

.. code-block::

    ase.calculators.kim.exceptions.KIMModelNotFound: Could not find model <> installed
    in any of the KIM API model collections on this system.
    See https://openkim.org/doc/usage/obtaining-models/ for instructions on installing models.

It usually happens when you have multiple kim-api installations, and the ASE/LAMMPS is not able to find the correct one.
Easiest solution can be to restart the terminal and load the correct conda environment.
If that does not help, try the following (solution i. is recommended),

i. using kim-api-activate

.. code-block:: shell

    source kim-api-activate

ii. set LD_LIBRARY_PATH manually

.. code-block:: shell

    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


Making changed permanent
-------------------------

.. attention::
    This can potentially mess up your conda install. So only do this if you understand the risks.
    **Also for MacOS you do not need the instructions below**, as anyway you are copying the libraries
    to the conda environment.

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
    echo "#\!/bin/bash" > "$CONDA_PREFIX/etc/conda/activate.d/env_activate.sh"
    echo "export _OLD_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH" >> "$CONDA_PREFIX/etc/conda/activate.d/env_activate.sh"
    echo "export _OLD_PATH=\$PATH" >> "$CONDA_PREFIX/etc/conda/activate.d/env_activate.sh"
    cat env.sh >> "$CONDA_PREFIX/etc/conda/activate.d/env_activate.sh"

    conda deactivate
    # Generate deactivation script
    echo "#\!/bin/bash" > "$env_prefix/etc/conda/deactivate.d/env_deactivate.sh"
    echo "export LD_LIBRARY_PATH=\$_OLD_LD_LIBRARY_PATH" >> "$env_prefix/etc/conda/deactivate.d/env_deactivate.sh"
    echo "export PATH=\$_OLD_PATH" >> "$env_prefix/etc/conda/deactivate.d/env_deactivate.sh"
    echo "unset _OLD_LD_LIBRARY_PATH" >> "$env_prefix/etc/conda/deactivate.d/env_deactivate.sh"
    echo "unset _OLD_PATH" >> "$env_prefix/etc/conda/deactivate.d/env_deactivate.sh"


Now everytime you activate the environment, it will automatically load the dependencies.
You can test this by opening a new terminal and running

.. code-block:: shell

    conda activate kliff_run
    lmp_serial -in in.lammps

