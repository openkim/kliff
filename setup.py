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
    return [get_pybind11_includes()]


def get_extra_compile_args():
    return ["-std=c++11"]


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

graph = Extension(
    "kliff.transforms.configuration_transforms.graph_module",
    sources=[
        "kliff/transforms/configuration_transforms/graphs/graph_generator_python_bindings.cpp",
        "kliff/transforms/configuration_transforms/graphs/neighbor_list.cpp",
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
    ext_modules=[sym_fn, bispectrum, neighlist, graph],
    install_requires=["requests", "scipy", "pyyaml", "monty", "loguru"],
    # TODO, use numpy<1.20 because ptemcee (or emcee) can only work with that.
    # Should we fork and fix it, and then remove the <1.20 requirement?
    # This also prohibits the use of py3.10; most distribution channels (e.g. conda)
    # do not have numpy<1.20 support anymore.
    extras_require={
        "test": ["pytest", "kimpy", "ptemcee", "emcee", "torch", "numpy<1.20"],
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
