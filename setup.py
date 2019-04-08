from setuptools import setup, Extension, find_packages
from distutils.sysconfig import get_config_vars
import os


# remove `-Wstrict-prototypes' that is for C not C++
cfg_vars = get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str and '-Wstrict-prototypes' in value:
        cfg_vars[key] = value.replace('-Wstrict-prototypes', '')


def get_extra_compile_args():
    return ['-std=c++11']


class get_pybind11_includes(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11 until it is actually
    installed, so that the ``get_include()`` method can be invoked.

    Borrowd from: https://github.com/pybind/python_example/blob/master/setup.py
    """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


symmetry_fn = Extension(
    'kliff.descriptors.symmetry_function.sf',
    sources=['kliff/descriptors/symmetry_function/sym_fn_bind.cpp',
             'kliff/descriptors/symmetry_function/sym_fn.cpp'],
    include_dirs=[get_pybind11_includes(),
                  get_pybind11_includes(user=True)],
    extra_compile_args=get_extra_compile_args(),
    language='c++',)

bispectrum = Extension(
    'kliff.descriptors.bispectrum.bs',
    sources=['kliff/descriptors/bispectrum/bispectrum.cpp',
             'kliff/descriptors/bispectrum/helper.cpp',
             'kliff/descriptors/bispectrum/bispectrum_bind.cpp'],
    include_dirs=[get_pybind11_includes(),
                  get_pybind11_includes(user=True)],
    extra_compile_args=get_extra_compile_args(),
    language='c++',)


def get_version(fname='kliff'+os.path.sep+'__init__.py'):
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


setup(name='kliff',
      version=get_version(),
      description='KLIFF interatomic potential fitting package',
      author='Mingjian Wen',
      url='https://github.com/mjwen/kliff',
      ext_modules=[symmetry_fn, bispectrum],
      # NOTE, subpackages need to be specified as well
      # packages=['kliff'],
      # NOTE, subpackages need to be included as well
      # packages=['kliff','tensorflow_op', 'geolm'],
      # package_dir={'geolm':'libs/geodesicLMv1.1/pythonInterface'},
      # package_data={'geolm':['_geodesiclm.so']},
      packages=find_packages(),
      scripts=kliff_scripts,
      install_requires=['scipy', 'pybind11'],
      zip_safe=False,
      )
