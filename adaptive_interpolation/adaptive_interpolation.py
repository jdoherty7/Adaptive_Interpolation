"""
Main methods used in the adaptive_interpolation library
"""
from __future__ import absolute_import

import numpy as np
import adaptive_interpolation.adapt as adapt
import adaptive_interpolation.approximator as app
import adaptive_interpolation.generate as generate
try:
    with_pickle = True
    import pickle
except:
    with_pickle = False


# takes an interval from a to b, a function, an interpolant order, and a
# maximum allowed error and returns an Approximator class representing
# a monomial interpolant that fits those parameters
def make_interpolant(a, b, func, order, error, basis="chebyshev",
                      adapt_type="Trivial", accurate=True):
    my_adapt = adapt.Interpolant(func, np.float64(order),
                                 np.float64(error), basis, accurate)
    my_adapt.run_adapt(np.float64(a), np.float64(b), adapt_type)
    return app.Approximator(my_adapt)


# Given an approximator class returned from a make an interpolant function,
# this function will return an approximator class that now contains C code
# to evaluate the interpolated function
# by default the code generated is non branching and vectorized. if the code
# is not vectorized then a domain_size must be given to the function
def generate_code(approx, branching=0, vectorized=1, domain_size=None):
    if (vectorized == 0) and (domain_size is None):
        string_err = "Please enter the number of points"
        string_err+= "that will be evaluated in domain_size."
        raise Exception(string_err)
    if approx.basis == 'monomial':
        if not branching:
            if vectorized:
                code = generate.gen_mono_v(approx)
            else:
                raise Exception("This code generation option currently unavailable.")
                # code = generate.gen_mono(approx, domain_size)
        else:
            if vectorized:
                raise Exception("This code generation option currently unavailable.")
                # code = generate.gen_mono_vb(approx)
            else:
                raise Exception("This code generation option currently unavailable.")
                # code = generate.gen_mono_b(approx, domain_size)

    elif approx.basis == 'chebyshev':
        if not branching:
            if vectorized:
                code = generate.gen_cheb_v(approx)
            else:
                raise Exception("This code generation option currently unavailable.")
                # code = generate.gen_cheb(approx, domain_size)
        else:
            if vectorized:
                raise Exception("This code generation option currently unavailable.")
                # code = generate.gen_cheb_vb(approx)
            else:
                raise Exception("This code generation option currently unavailable.")
                # code = generate.gen_cheb_b(approx, domain_size)

    elif approx.basis == 'legendre':
        if not branching:
            if vectorized:
                #raise Exception("This code generation option currently unavailable.")
                code = generate.gen_leg_v(approx)
            else:
                raise Exception("This code generation option currently unavailable.")
                # code = generate.gen_leg(approx, domain_size)
        else:
            if vectorized:
                raise Exception("This code generation option currently unavailable.")
                # code = generate.gen_mono_vb(approx)
            else:
                raise Exception("This code generation option currently unavailable.")
                # code = generate.gen_leg_b(approx, domain_size)
    approx.code = code
    return code


# save approximator class to the given path
def save_approximation(path_name, approximation):
    if with_pickle:
        my_file = open(path_name, "w")
        pickle.dump(approximation, my_file)
        my_file.close()
    else:
        raise Exception("Save requires pickle module be installed")


# load approximator class from the given path
def get_saved_approximation(path_name):
    if with_pickle:
        my_file = open(path_name, "r")
        app = pickle.load(my_file)
        my_file.close()
        return app
    else:
        raise Exception("Load requires pickle module be installed")


# code is a string of C code to be evaluated. x is a numpy array
# that is a type float64 and is in the interval specified by the user
# upon the creation of the interpolant. if the code is not vectorized
# then the approximator class must also be given to the function
def run_code(x, approx, vectorized=True):
    if approx.code == 0:
        string_err = "Approximator class does not have any associated "
        string_err+= "code. Run a generate method to add code to the class."
        raise Exception(string_err)
    if vectorized and (approx.basis == 'chebyshev' or approx.basis == 'legendre'):
        return generate.run_ortho_vec(x, approx)
    elif vectorized and approx.basis == 'monomial':
        return generate.run_mono_vec(x, approx)
    raise Exception("Non vectorized is not currently supported. \
                     Please choose a vectorized method.")

