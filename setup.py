from Cython.Distutils import build_ext
from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize
import numpy

compile_args = [
        '-O3',
        # '-ffast-math',  # could be kind of unsafe
        ]

ext_modules = [
        Extension('c_rot_steps',
                  sources=['c_rot_steps.pyx'],
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=compile_args,
                  language='C')
        ]

setup(name='c_rot_steps',
      description='Fast Rot Steps',
      ext_modules=cythonize(ext_modules),
      cmdclass={'build_ext':build_ext}
      )
