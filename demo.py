"""
Demonstration of capabilities of the module

"""
from __future__ import absolute_import

import time
import numpy as np
import numpy.linalg as la
import scipy.special as spec
import matplotlib.pyplot as plt
import adaptive_interpolation.adapt as adapt
import adaptive_interpolation.generate as generate
import adaptive_interpolation.approximator as app
import adaptive_interpolation.adaptive_interpolation as adapt_i

try:
    with_pyopencl = True
    import pyopencl
except:
    with_pyopencl = False


# bessel function for testing
def f(x):
    return spec.jn(0, x)


# a function for testing
def f1(x0):
    xs = []
    for x in x0:
        if x < 1:
            xs.append(1 + x)
        elif (1 <= x) and (x < 2.02):
            xs.append(1 + x**2)
        elif (2.02 <= x) and (x < 3.5):
            xs.append(-3*np.log(x))
        elif (3.5 <= x) and (x < 4.4):
            xs.append(np.exp(np.sqrt(x)))
        elif (4.4 <= x) and (x < 7.001):
            xs.append(3)
        elif (7.001 <= x) and (x < 9.306):
            xs.append(np.sqrt(x**4.4) / 100.)
        elif (9.306 <= x) and (x <= 11):
            xs.append(x - 3)
    return np.array(xs)


# plot the absolute errors as well as the actual and approximated functions
def my_plot(x, actual, approximation, abs_errors, allowed_error, ap):
    plt.figure()
    plt.title('Actual and Approximate values Graphed')
    t, = plt.plot(x, actual, 'r', label='True Values')
    e, = plt.plot(x, approximation, 'b', label='Interpolated Values')
    plt.legend(handles=[t, e], loc=0)

    plt.figure()
    plt.yscale('log')
    plt.title('Absolute Error in Interpolated Values')
    a, = plt.plot(x, abs_errors+1e-17, 'g', label='Absolute Errors')
    b, = plt.plot(x, 0*x + allowed_error, 'r', label='Maximum Allowed Relative Error')
    all_ranges = []
    for my_range in ap.ranges:
        all_ranges.append(my_range[0])
        all_ranges.append(my_range[1])
    for val in list(set(all_ranges)):
        plt.axvline(x=val)
    c, = plt.plot(ap.used_midpoints, ap.rel_errors, 'bs', label='Relative Errors')
    plt.legend(handles=[a, b, c], loc=0, fontsize='small')
    plt.show()


# This will demo the capabilities of interpolating a function with a fixed order method
# basis is a string specifying your basis. function is the given function to interpolate
# allowed error is the maximum relative error allowed on the entire interval.
def demo_adapt(function, order, allowed_error, basis,
               accurate=True, a=0, b=10):

    print("Creating Interpolant")
    if basis == 'chebyshev':
        my_approx = adapt_i.make_chebyshev_interpolant(a, b, 
                                         function, order, allowed_error,
                                         False, accurate)
    elif basis == 'legendre':
        my_approx = adapt_i.make_legendre_interpolant(a, b, 
                                        function, order, allowed_error,
                                        False, accurate)
    else:
        my_approx = adapt_i.make_monomial_interpolant(a, b,
                                        function, order, allowed_error,
                                        False, accurate)
    print("Generating Code")
    code = adapt_i.generate_code(my_approx)
    print(code)
    print("Evaluating Interpolant")
    x = np.linspace(a, b, 1e4, dtype=np.float64)
    if with_pyopencl: 
        est = adapt_i.run_code(code, x, my_approx)
    else: 
        est = my_approx.evaluate(x)
    print("Evaluating Function")
    true = function(x)
    print("Plotting")
    my_plot(x, true, est, abs(true-est), allowed_error, my_approx)


# This will demo the capabilities of interpolating a function with a variable order method
# this will run order times slower than the normal adaptive method. 
# basis is a string specifying your basis. function is the given function to interpolate
# allowed error is the maximum relative error allowed on the entire interval.
def demo_adapt_variable(function, order, allowed_error, basis, 
                        accurate=True, a=0, b=10):

    print("Creating Interpolant")
    if basis == 'chebyshev':
        my_approx = adapt_i.make_chebyshev_interpolant(a, b, 
                                         function, order, allowed_error, 
                                         True, accurate)
    elif basis == 'legendre':
        my_approx = adapt_i.make_legendre_interpolant(a, b, 
                                        function, order, allowed_error,
                                        True, accurate)
    else:
        my_approx = adapt_i.make_monomial_interpolant(a, b,
                                        function, order, allowed_error,
                                        True, accurate)
    print("Generating Code")
    code = adapt_i.generate_code(my_approx)
    print(code)
    print("Evaluating Interpolant")
    x = np.linspace(a, b, 5e4, dtype=np.float64)
    if with_pyopencl: 
        est = adapt_i.run_code(code, x, my_approx)
    else: 
        est = my_approx.evaluate(x)
    print("Evaluating Function")
    true = function(x)
    print("Plotting")
    my_plot(x, true, est, abs(true-est), allowed_error, my_approx)


# run the main program
if __name__ == "__main__":
    # method interpolates a bessel function
    demo_adapt(f, 10, 1e-13, 'chebyshev')
    # method interpolating a exact function
    my_f = lambda x: np.sin(np.float64(1.)/x)
    demo_adapt(my_f, 20, 1e-3, 'chebyshev', a=.03, b=1)
    # variable order interpolation method
    demo_adapt_variable(f1, 10, 1e-5, 'monomial')

