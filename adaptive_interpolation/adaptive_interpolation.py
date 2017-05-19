"""
Main methods used in the adaptive_interpolation library
"""
from __future__ import absolute_import

import numpy as np
import adaptive_interpolation.adapt as adapt
import adaptive_interpolation.approximator as app
import adaptive_interpolation.generate as generate



# takes an interval from a to b, a function, an interpolant order, and a
# maximum allowed error and returns an Approximator class representing
# a monomial interpolant that fits those parameters
def make_interpolant(a, b, func, order, error, basis="chebyshev",
                      adapt_type="Trivial", dtype='64', accurate=True):
    my_adapt = adapt.Interpolant(func, order, error, basis, dtype, accurate)
    my_adapt.run_adapt(a, b, adapt_type)
    approximation = app.Approximator(my_adapt)
    dt = int(dtype)
    if dt <=32:
        approximation.dtype = "float"
    elif dt <= 64:
        approximation.dtype = "double"
    elif dt <= 80:
        approximation.dtype = "long double"
    else:
        raise Exception("Incorrect data type specified")
    return approximation


# Given an approximator class returned from a make an interpolant function,
# this function will return an approximator class that now contains C code
# to evaluate the interpolated function
# by default the code generated is non branching and vectorized. if the code
# is not vectorized then a domain_size must be given to the function
def generate_code(approx, size=None, vector_width=None):
    if ((vector_width is not None) and (size is None)) \
    or ((vector_width is None) and (size is not None)):
        string_err = "Both vector width and domain_size must be given "
        string_err+= "if running single threaded code."
        raise Exception(string_err)

    approx.vector_width = vector_width
    approx.size = size
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


# use to save the approximator class for later use
def write_to_file(file_path, approx):
    cd_file = open(file_path + "_code.txt", "w")
    bs_file = open(file_path + "_basis.txt", "w")
    cd_file.write(approx.code)
    bs_file.write(approx.basis)
    np.save(file_path + "_run_vector", approx.run_vector)
    cd_file.close()
    bs_file.close()

# loads a new approximator class with the variables saved at the file path
def load_from_file(file_path):
    try:
        cd_file = open(file_path + "_code.txt", "r")
        bs_file = open(file_path + "_basis.txt", "r")
        approx = Approximator()
        approx.code = cd_file.read()
        approx.basis = bs_file.read()
        approx.run_vector = np.load(file_path + "_run_vector.npy")
        cd_file.close()
        bs_file.close()
        return approx
    except:
        str_err = "Specified file {0}, does not exist".format(file_path)
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
        knl, q, x, y, tree = generate.build_code(x, approx)
        run_time, output = generate.run_single(knl, q, x, y, tree, vector_width)
    return run_time, output


