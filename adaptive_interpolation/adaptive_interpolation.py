"""
Main methods used in the adaptive_interpolation library
"""
from __future__ import absolute_import

import pickle
import numpy as np
import adaptive_interpolation.adapt as adapt
import adaptive_interpolation.approximator as app
import adaptive_interpolation.generate as generate



def make_interpolant(a, b, func, order, error, basis="chebyshev",
                      adapt_type="Remez", dtype='64', accurate=True, optimizations=[]):
    """
    Takes an interval from a to b, a function, an interpolant order, and a
    maximum allowed error and returns an Approximator class representing
    a monomial interpolant that fits those parameters
    """

    my_adapt = adapt.Interpolant(func, order, error, basis, dtype, accurate, optimizations=optimizations)
    my_adapt.run_adapt(a, b, adapt_type)
    approximation = app.Approximator(my_adapt, optimizations=optimizations)
    dt = int(dtype)
    if dt <= 32:
        approximation.dtype_name = "float"
    elif dt <= 64:
        approximation.dtype_name = "double"
    elif dt <= 80:
        approximation.dtype_name = "long double"
    else:
        raise Exception("Incorrect data type specified")
    return approximation


def generate_code(approx, size=None, vector_width=1, cpu=False):
    """
    Given an approximator class returned from a make an interpolant function,
    this function will return an approximator class that now contains C code
    to evaluate the interpolated function
    by default the code generated is non branching and vectorized. if the code
    is not vectorized then a domain_size must be given to the function
    if size is not given then code is parallelized, Needed if GPU only using one global group
    if vector width is given then the code is vectorized, only important for GPU
    """
    approx.vector_width = vector_width
    approx.size = size
    if cpu:
        if size is None:
            raise Exception("The size must be known to generate cpu code.")

        if approx.basis != 'chebyshev':
            string_err = "ISPC generation only compatible with "
            string_err += "chebyshev interpolants currently."
            raise Exception(string_err)
        else:
            code = generate.gen_ispc(approx)
    else:
        if ((vector_width != 1) and (size is None)) \
        or ((vector_width == 1) and (size is not None)):
            string_err = "Both vector width and domain_size must be given "
            string_err+= "if running single-threaded code."
            raise Exception(string_err)

        if approx.basis == 'monomial':
            code = generate.gen_mono(approx)
        elif approx.basis == 'chebyshev':
            code = generate.gen_cheb(approx)
        elif approx.basis == 'legendre':
            code = generate.gen_leg(approx)
        else:
            raise Exception("Incorrect basis provided: monomial, chebyshev, legendre.")
    approx.code = code
    return code


def write_to_file(file_path, approx):
    """
    file_path      : File path to save instance to
    type file_path : str
    approx         : Instance to be saved using pickle
    type approx    : Approximator object

    Use to save the approximator class for later use
    """
    with open(file_path, 'wb') as f:
        pickle.dump(approx, f, pickle.HIGHEST_PROTOCOL)

 
def load_from_file(file_path):
    """
    file_path      : File path to save instance to
    type file_path : str
    ret            : Instance to be saved using pickle
    type ter       : Approximator object

    Loads a new approximator class with the variables saved at the file path
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        str_err = "Error loading instance from {0}".format(file_path)
        raise Exception(str_err)


# x is evaluated according to given approximation class.
# if vector_width is given to generative method then the
# code is run single threaded with specified width
def run_approximation(x, approx):
    if approx.code == 0:
        string_err = "Approximator class does not have any associated "
        string_err+= "code. Run a generate method to add code to the class."
        raise Exception(string_err)

    if approx.vector_width is None:
        output = generate.run(x, approx)
        run_time = None
    else:
        knl, q, tree = generate.build_code(approx)
        run_time, output = generate.run_single(x, approx)
    return run_time, output


