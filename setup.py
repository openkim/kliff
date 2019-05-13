from setuptools import setup, Extension, find_packages
from distutils.sysconfig import get_config_vars
import os


# remove `-Wstrict-prototypes' that is for C not C++
cfg_vars = get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str and '-Wstrict-prototypes' in value:
        cfg_vars[key] = value.replace('-Wstrict-prototypes', '')


class get_pybind11_includes:
    """Helper class to determine the pybind11 include path.

    The purpose of this class is to postpone importing pybind11 until it is
    actually installed, so that the ``get_include()`` method can be invoked.

    See:
    https://github.com/pybind/python_example/blob/master/setup.py
    https://github.com/pybind/python_example/issues/32
    """

    def __init__(self, user=False):
        try:
            import pybind11
        except ImportError:
            if subprocess.call([sys.executable, '-m', 'pip', 'install', 'pybind11']):
                raise RuntimeError('pybind11 install failed.')
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


def get_includes():
    return [get_pybind11_includes(), get_pybind11_includes(user=True)]


def get_extra_compile_args():
    return ['-std=c++11']


sym_fn = Extension(
    'kliff.descriptors.symmetry_function.sf',
    sources=[
        'kliff/descriptors/symmetry_function/sym_fn_bind.cpp',
        'kliff/descriptors/symmetry_function/sym_fn.cpp',
    ],
    include_dirs=get_includes(),
    extra_compile_args=get_extra_compile_args(),
    language='c++',
)

bispectrum = Extension(
    'kliff.descriptors.bispectrum.bs',
    sources=[
        'kliff/descriptors/bispectrum/bispectrum.cpp',
        'kliff/descriptors/bispectrum/helper.cpp',
        'kliff/descriptors/bispectrum/bispectrum_bind.cpp',
    ],
    include_dirs=get_includes(),
    extra_compile_args=get_extra_compile_args(),
    language='c++',
)


def get_version(fname=os.path.join('kliff', '__init__.py')):
    with open(fname) as fin:
        for line in fin:
            line = line.strip()
            if '__version__' in line:
                v = line.split('=')[1]
                # stripe white space, and ' or " in string
                if "'" in v:
                    version = v.strip("' ")
                elif '"' in v:
                    version = v.strip('" ')
                break
    return version


kliff_scripts = ['bin/kliff']


setup(
    name='kliff',
    version=get_version(),
    # NOTE, subpackages need to be specified as well
    # packages=['kliff'],
    # NOTE, subpackages need to be included as well
    # packages=['kliff','tensorflow_op', 'geolm'],
    # package_dir={'geolm':'libs/geodesicLMv1.1/pythonInterface'},
    # package_data={'geolm':['_geodesiclm.so']},
    packages=find_packages(),
    ext_modules=[sym_fn, bispectrum],
    scripts=kliff_scripts,
    install_requires=['scipy', 'pybind11', 'pytest'],
    author='Mingjian Wen',
    author_email='wenxx151@gmail.com',
    url='https://github.com/mjwen/kliff',
    description='KLIFF: KIM-based Learning-Integrated Fitting Framework',
    long_description='KLIFF: KIM-based Learning-Integrated Fitting Framework',
    classifiers=(
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: Common Development and Distribution License 1.0 (CDDL-1.0)',
        'Operating System :: OS Independent',
    ),
    zip_safe=False,
)
