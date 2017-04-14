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

