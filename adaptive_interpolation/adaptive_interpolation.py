"""
Main methods used in the adaptive_interpolation library
"""
import adapt
import generate
import numpy as np
import approximator as app


# takes an interval from a to b, a function, an interpolant order, and a
# maximum allowed error and returns an Approximator class representing
# a monomial interpolant that fits those parameters
def make_monomial_interpolant(a, b, func, order, error, variable=False):
    # node type used random and cheb are options, otherwise equispaced is used
    nt = 'chebyshev'
    # chebyshev, legendre, or monomials
    interp_choice = 'monomial'

    adapt_class = adapt.adaptive(func, a, b, error, nt, order,
                                 interp_choice, variable)
    return app.Approximator(adapt_class)


# takes an interval from a to b, a function, an interpolant order, and a
# maximum allowed error and returns an Approximator class representing
# a chebyshev interpolant that fits those parameters
def make_chebyshev_interpolant(a, b, func, order, error, variable=False):
    # interpolant parameters
    # node type used random and cheb are options, otherwise equispaced is used
    nt = 'chebyshev'
    # chebyshev, legendre, or monomials
    interp_choice = 'chebyshev'

    adapt_class = adapt.adaptive(func, a, b, error, nt, order,
                                 interp_choice, variable)
    return app.Approximator(adapt_class)


# takes an interval from a to b, a function, an interpolant order, and a
# maximum allowed error and returns an Approximator class representing
# a legendre interpolant that fits those parameters
def make_legendre_interpolant(a, b, func, order, error, variable=False):
    # interpolant parameters
    # node type used random and cheb are options, otherwise equispaced is used
    nt = 'chebyshev'
    # chebyshev, legendre, or monomials
    interp_choice = 'legendre'

    adapt_class = adapt.adaptive(func, a, b, error, nt, order,
                                 interp_choice, variable)
    return app.Approximator(adapt_class)


# Given an approximator class returned from a make an interpolant function,
# this function will return an approximator class that now contains C code
# to evaluate the interpolated function
# by default the code generated is non branching and vectorized. if the code
# is not vectorized then a domain_size must be given to the function
def generate_code(approx, branching=0, vectorized=1, domain_size=None):
    if (vectorized == 0) and (domain_size is None):
        print("Please enter the number of points that will \
               be evaluated in domain_size.")
        return 0
    if approx.basis == 'monomials':
        if not branching:
            if vectorized:
                code = generate.gen_mono_v(approx)
            else:
                pass
                # code = generate.gen_mono(approx, domain_size)
        else:
            if vectorized:
                code = generate.gen_mono_vb(approx)
            else:
                code = generate.gen_mono_b(approx, domain_size)

    elif approx.basis == 'chebyshev':
        if not branching:
            if vectorized:
                code = generate.gen_cheb_v(approx)
            else:
                pass
                # code = generate.gen_cheb(approx, domain_size)
        else:
            if vectorized:
                pass
                # code = generate.gen_cheb_vb(approx)
            else:
                pass
                # code = generate.gen_cheb_b(approx, domain_size)

    elif approx.basis == 'legendre':
        if not branching:
            if vectorized:
                pass
                # code = generate.gen_leg_v(approx)
            else:
                pass
                # code = generate.gen_leg(approx, domain_size)
        else:
            if vectorized:
                pass
                # code = generate.gen_mono_vb(approx)
            else:
                code = generate.gen_leg_b(approx, domain_size)
    approx.code = code
    return code


# use to save the generated code for later use
# NOTE: this only works if run from main directory, where
# the folder generated_code exists
def save_code(file_name, code):
    my_file = open("generated_code/"+file_name+".txt", "w")
    my_file.write(code)
    my_file.close()


# get a string of the C code that was previously saved
# NOTE: this only works if run from main directory, where
# the folder generated_code exists
def get_saved_code(file_name):
    my_file = open("generated_code/"+file_name+".txt", "r")
    # code is just on first line of file, so get this then run it
    code = my_file.readline()
    return code


# code is a string of C code to be evaluated. x is a numpy array
# that is a type float64 and is in the interval specified by the user
# upon the creation of the interpolant. if the code is not vectorized
# then the approximator class must also be given to the function
def run_code(code, x, approx=0, vectorized=True):
    if vectorized and approx != 0:
        return generate.run_c_v(x, approx, code)
    elif not vectorized:
        return generate.run_c(x, code)
    print("You must give an appropriate appxoimator class if the \
           code is not vectorized.")
    return 0
