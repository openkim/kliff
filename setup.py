from distutils.sysconfig import get_config_vars
from pathlib import Path

from setuptools import Extension, find_packages, setup

# remove `-Wstrict-prototypes' that is for C not C++
cfg_vars = get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str and "-Wstrict-prototypes" in value:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")


class get_pybind11_includes:
    """
    Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11 until it is actually
    installed, so that the ``get_include()`` method can be invoked.

    see:
    https://github.com/pybind/python_example/blob/master/setup.py
    https://github.com/pybind/python_example/issues/32
    """

    def __str__(self):
        import pybind11

        return pybind11.get_include()


def get_includes():
    return [get_pybind11_includes(), "kliff/neighbor"]


def get_extra_compile_args():
    return ["-std=c++11"]


# TODO: explore -Ofast and -march=native

sym_fn = Extension(
    "kliff.legacy.descriptors.symmetry_function.sf",
    sources=[
        "kliff/legacy/descriptors/symmetry_function/sym_fn_bind.cpp",
        "kliff/legacy/descriptors/symmetry_function/sym_fn.cpp",
        "kliff/legacy/descriptors/symmetry_function/helper.cpp",
    ],
    include_dirs=get_includes(),
    extra_compile_args=get_extra_compile_args(),
    language="c++",
)

bispectrum = Extension(
    "kliff.legacy.descriptors.bispectrum.bs",
    sources=[
        "kliff/legacy/descriptors/bispectrum/bispectrum_bind.cpp",
        "kliff/legacy/descriptors/bispectrum/bispectrum.cpp",
        "kliff/legacy/descriptors/bispectrum/helper.cpp",
    ],
    include_dirs=get_includes(),
    extra_compile_args=get_extra_compile_args(),
    language="c++",
)

neighlist = Extension(
    "kliff.neighbor.neighlist",
    sources=[
        "kliff/neighbor/neighbor_list.cpp",
        "kliff/neighbor/neighbor_list_bind.cpp",
    ],
    include_dirs=get_includes(),
    extra_compile_args=get_extra_compile_args(),
    language="c++",
)

graph_module = Extension(
    "kliff.transforms.configuration_transforms.graphs.graph_module",
    sources=[
        "kliff/transforms/configuration_transforms/graphs/radial_graph.cpp",
        "kliff/neighbor/neighbor_list.cpp",
    ],
    include_dirs=get_includes(),
    extra_compile_args=get_extra_compile_args(),
    language="c++",
)


def get_version():
    fname = Path(__file__).parent.joinpath("kliff", "__init__.py")
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if "__version__" in line:
                v = line.split("=")[1]
                # stripe white space, and ' or " in string
                if "'" in v:
                    version = v.strip("' ")
                elif '"' in v:
                    version = v.strip('" ')
                break
    return version


def get_readme():
    fname = Path(__file__).parent.joinpath("README.md")
    with open(fname, "r") as f:
        readme = f.read()
    return readme


setup(
    name="kliff",
    version=get_version(),
    packages=find_packages(),
    ext_modules=[sym_fn, bispectrum, neighlist, graph_module],
    install_requires=[
        "requests",
        "scipy",
        "pyyaml",
        "monty",
        "loguru",
        "ase<3.23",
        "numpy<2.0",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "kimpy",
            "emcee",
            # The below one works if one installs this repo from source; however,
            # PyPI does not allow this syntax. So we comment it out and a user need to
            # install it manually for now.
            # "ptemcee @ git+https://github.com/yonatank93/ptemcee.git@enhance_v1.0.0",
            "numpy<2.0",
            "ase<3.23",
            "libdescriptor",
            "lmdb",
        ],
        "torch": [
            "torch",
            "torch_geometric",
            "pytorch_lightning",
            "torch_scatter",
            "tensorboard",
            "tensorboardx",
        ],
        "docs": [
            "sphinx",
            "furo",
            "myst-nb",
            "sphinx-autodoc-typehints",
            "sphinx-copybutton",
            "matplotlib",
        ],
    },
    entry_points={"console_scripts": ["kliff = kliff.cmdline.cli:main"]},
    author="Mingjian Wen",
    author_email="wenxx151@gmail.com",
    url="https://github.com/openkim/kliff",
    description="KLIFF: KIM-based Learning-Integrated Fitting Framework",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
