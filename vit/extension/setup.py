from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

curdir = os.getcwd()
setup(name = 'fused_bert',
      version = '0.1',
      ext_modules = [cpp_extension.CppExtension(
            name = 'fused_bert',
            sources = ['bert.cpp'],
            include_dirs = [curdir + '/mklml/include', curdir + '/onednn_helper'],
            library_dirs = [curdir + '/mklml/lib', curdir],
            runtime_library_dirs = [curdir + '/mklml/lib', curdir],
            depends = [curdir + '/mklml/lib/libiomp5.so', ],
            #define_macros=[('DEBUG', 1), ('USE_MKL', 1)],
            #define_macros=[('DEBUG', 1), ],
            define_macros=[('USE_ONEDNN', 1), ],
            extra_compile_args = ['-fopenmp', '-mavx512f', '-mavx512bw', '-mavx512vl', '-mfma', '-std=c++17'],
            extra_link_args = ['-l:libsgemm.a', '-l:libigemm.a', '-liomp5', '-lmklml_intel', '-ldnnl', ]),
      ],
      cmdclass = {'build_ext': cpp_extension.BuildExtension})

