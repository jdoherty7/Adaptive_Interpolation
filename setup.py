try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description' : 'Interpolation of a function \
     and generates C code to quickly evaluate \
     the interpolation using pyopencl.',
    'author' : 'John Doherty',
    'author_email' : 'muons.quarks@gmail.com',
    'version' : '0.2',
    'license' : 'MIT',
    'url' : 'https://www.github.com/jdoherty7/Adaptive_Interpolation',
    'install_requires': ['nose', 'numpy', 'matplotlib',
    'pyopencl', 'scipy', 'cgen'],
    'packages': ['adaptive_interpolation'],
    'name': 'adaptive_interpolation'
}


setup(**config)

