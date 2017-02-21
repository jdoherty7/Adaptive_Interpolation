try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description' : 'Creates an interpolation of a function \
     on a given domain and returns C code to quickly evaluate \
     said interpolation using pyopencl.',
    'author' : 'John Doherty',
    'version' : '0.1',
    'license' : 'MIT',
    'install_requires': ['nose', 'numpy', 'matplotlib', \
    'pyopencl', 'scipy', 'cgen'],
    'packages': ['adaptive_interpolation', 'tests'],
    'name': 'adaptive_interpolation'
}


setup(**config)

