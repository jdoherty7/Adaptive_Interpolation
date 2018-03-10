try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description' : 'Create and evaluate vectorized function approximations',
    'author' : 'John Doherty',
    'author_email' : 'jjdoher2@illinois.edu',
    'version' : '2017.0.4',
    'license' : 'MIT',
    'url' : 'https://www.github.com/jdoherty7/Adaptive_Interpolation',
    'install_requires': ['numpy', 'matplotlib', 'pyopencl', 'scipy', 'cgen'],
    'packages': ['adaptive_interpolation'],
    'scripts': ['demo.py'],
    'name': 'adaptive-interpolation'
}

setup(**config)

