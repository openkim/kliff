from setuptools import setup, Extension
from distutils.sysconfig import get_config_vars
import os
import numpy


# remove `-Wstrict-prototypes' that is for C not C++
cfg_vars = get_config_vars()
for key, value in cfg_vars.items():
  if type(value) == str and '-Wstrict-prototypes' in value:
     cfg_vars[key] = value.replace('-Wstrict-prototypes', '')


def tf_includes():
  try:
    import tensorflow as tf
    return tf.sysconfig.get_include()
  except ImportError:
    raise ImportError('tensorflow is not found. install it first.')


def tf_extra_compile_args():
  args = ['-std=c++11', '-Wall', '-O2', '-fPIC']
  # gcc 5 needs the following
  args_gcc5 = ['-D_GLIBCXX_USE_CXX11_ABI=0']
  return args + args_gcc5


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


tf_module = Extension('tensorflow_op.int_pot_op',
   sources = ['tensorflow_op/int_pot_op.cpp'],
   include_dirs = [tf_includes()],
   #library_dirs = [],
   libraries = ['m'],
   extra_compile_args = tf_extra_compile_args(),
   #extra_link_args = [],
   language = 'c++',
    )


desc_module = Extension('desc',
    sources = ['openkim_fit/descriptor_bind.cpp', 'openkim_fit/descriptor_c.cpp'],
    include_dirs = [get_pybind11_includes(), get_pybind11_includes(user=True)],
    extra_compile_args = get_extra_compile_args(),
    language = 'c++',
    )


setup(name='openkim_fit',
    version='0.0.1',
    description='OpenKIM based interatomic potential fitting program.',
    author='Mingjian Wen',
    url='https://openkim.org',
    packages=['openkim_fit','tensorflow_op', 'geolm'],
    package_dir={'geolm':'libs/geodesicLMv1.1/pythonInterface'},
    package_data={'geolm':['_geodesiclm.so']},
    ext_modules=[tf_module, desc_module],
    install_requires = ['scipy'],
    setup_requires = ['numpy', 'pybind11>=2.2'],
    zip_safe = False,
    )


