try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description' : 'Used to adaptively interpolate a function \
     then generates C code to quickly evaluate \
     the interpolation using pyopencl.',
    'author' : 'John Doherty',
    'author_email' : 'muons.quarks@gmail.com',
    'version' : '0.3',
    'license' : 'MIT',
    'url' : 'https://www.github.com/jdoherty7/Adaptive_Interpolation',
    'install_requires': ['nose', 'numpy', 'matplotlib',
    'pyopencl', 'scipy', 'cgen'],
    'packages': ['adaptive_interpolation'],
    'scripts': ['demo.py'],
    'name': 'adaptive-interpolation'
}


setup(**config)

