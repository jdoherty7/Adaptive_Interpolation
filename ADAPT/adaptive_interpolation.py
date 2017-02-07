"""
Main methods used in the adaptive_interpolation library
2/7/17
"""
import generate
import numpy as np
import adapt
import approximator as app



# takes a domain, a function, an interpolant order, and a maximum
# allowed error and returns an Approximator class representing
# a monomial interpolant that fits those parameters
def make_monomial_interpolant(a, b, func, order, error):
    # node type used random and cheb are options, otherwise equispaced is used
    nt = 'chebyshev'
    # chebyshev, legendre, or monomials
    interp_choice = 'monomial'

    adapt_class = adapt.adaptive(func, a, b, error, nt, order, interp_choice)
    return app.Approximator(adapt_class)


# takes a domain, a function, an interpolant order, and a maximum
# allowed error and returns an Approximator class representing
# a chebyshev interpolant that fits those parameters
def make_chebyshev_interpolant(a, b, func, order, error):
    # interpolant parameters
    # node type used random and cheb are options, otherwise equispaced is used
    nt = 'chebyshev'
    # chebyshev, legendre, or monomials
    interp_choice = 'chebyshev'

    adapt_class = adapt.adaptive(func, a, b, error, nt, order, interp_choice)
    return app.Approximator(adapt_class)


# takes a domain, a function, an interpolant order, and a maximum
# allowed error and returns an Approximator class representing
# a legendre interpolant that fits those parameters
def make_legendre_interpolant(a, b, func, order, error):
    # interpolant parameters
    # node type used random and cheb are options, otherwise equispaced is used
    nt = 'chebyshev'
    # chebyshev, legendre, or monomials
    interp_choice = 'legendre'

    adapt_class = adapt.adaptive(func, a, b, error, nt, order, interp_choice)
    return app.Approximator(adapt_class)


# generate code. given an approximator class will return a string
# representing executable C code that can be run using pyopencl
# this is done by using the correct run method described
def generate_code(approx, branching, vectorized, domain_size=None):
    
    if approx.basis == 'monomials':
        if not branching:
            if vectorized:
                code = generate.gen_mono_v(approx)
            else:
                pass
                #code = generate.gen_mono(approx, domain_size)
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
                #code = generate.gen_leg(approx, domain_size)
        else:
            if vectorized:
                pass
                # code = generate.gen_mono_vb(approx)
            else:
                code = generate.gen_leg_b(approx, domain_size)
    approx.code = code
    return code


# use to save the generated code for later use
def save_code(file_name, code):
    my_file = open("generated_code/"+file_name+".txt", "w")
    my_file.write(code)
    my_file.close()


# get a string of the C code that was previously saved
def get_saved_code(file_name):
    my_file = open("generated_code/"+file_name+".txt", "r")
    # code is just on first line of file, so get this then run it
    code = my_file.readline()
    return code


# run code, must specify if vectorized code or not
def run_code(code, approx, domain_size, vectorized):
    a = approx.heapp[-1][0]
    b = approx.heap[-1][0]
    x = np.linspace(a, b, domain_size).astype(np.float64)
    if vectorized:
        generate.run_c_v(x, code)
    else:
        generate.run_c(x, approx.midpoints, approx.coeff, code)
