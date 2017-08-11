from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy



# to use geodesiclm add the following:
# from geolm.geodesiclm import geodesiclm
# to use openkim modules, such as kimcalculator.py, add
# from openkim_kim.kimcalculator import KIMcalculator
setup(name='openkim_fit',
      version='0.0.1',
      description='OpenKIM based interatomic potential fitting program.',
      author='Mingjian Wen',
      url='https://openkim.org',
      packages=['openkim_fit','geolm','tfop'],
      package_dir={'geolm':'libs/geodesicLMv1.1/pythonInterface',
                   'tfop':'libs/tensorflowOp'},
      package_data={'geolm':['_geodesiclm.so'],
                    'tfop':['int_pot_op.so']},
      ext_modules=cythonize([Extension('desc',
      sources=['openkim_fit/desc.pyx', 'openkim_fit/descriptor_c.cpp'],
      language = 'c++',
      include_dirs=[numpy.get_include()] )]),
     )

# NOTE
# It seems not easy to not install geodesiclm in a package.

#from setuptools import setup,find_packages

# if we use setuptools, _geodesiclm.so will be egged, then it is not each to
# import it. People who write code need to use pkg_resources to get access to
# it. It is not easy to do.


