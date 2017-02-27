"""
Demonstration of capabilities of the module

"""

import time
import numpy as np
import numpy.linalg as la
import scipy.special as spec
import matplotlib.pyplot as plt
import adaptive_interpolation.adaptive_interpolation as adapt
import adaptive_interpolation.adaptive_interpolation as generate
import adaptive_interpolation.adaptive_interpolation as app
import adaptive_interpolation.adaptive_interpolation as adapt_i


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
    for i in range(len(ap.ranges)):
        plt.axvline(x=ap.ranges[i][0])
    plt.axvline(x=ap.ranges[i][1])
    c, = plt.plot(ap.used_midpoints, ap.rel_errors, 'bs', label='Relative Errors')
    plt.legend(handles=[a, b, c], loc=0, fontsize='small')
    plt.show()


# This will demo the capabilities of interpolating a function with a fixed order method
# basis is a string specifying your basis. function is the given function to interpolate
# allowed error is the maximum relative error allowed on the entire interval.
def demo_adapt(function, order, allowed_error, basis,
               with_pyopencl=False, accurate=True):
    a, b = 0, 10
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
    code = adapt_i.generate_code(my_approx)
    print(code)
    print("Evaluating Interpolant")
    x = np.linspace(a, b, 1e3, dtype=np.float64)
    if with_pyopencl: 
        est = adapt_i.run_code(code, x, my_approx)
    else: 
        est = my_approx.evaluate(x)
    print("Evaluating Function")
    true = function(x)
    my_plot(x, true, est, abs(true-est), allowed_error, my_approx)


# This will demo the capabilities of interpolating a function with a variable order method
# this will run order times slower than the normal adaptive method. 
# basis is a string specifying your basis. function is the given function to interpolate
# allowed error is the maximum relative error allowed on the entire interval.
def demo_adapt_variable(function, order, allowed_error, basis, 
                        with_pyopencl=False, accurate=True):
    a, b = 0, 10
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
    code = adapt_i.generate_code(my_approx)
    print(code)
    print("Evaluating Interpolant")
    x = np.linspace(a, b, 1e3, dtype=np.float64)
    if with_pyopencl: 
        est = adapt_i.run_code(code, x, my_approx)
    else: 
        est = my_approx.evaluate(x)
    print("Evaluating Function")
    true = function(x)
    my_plot(x, true, est, abs(true-est), allowed_error, my_approx)


# run the main program
if __name__ == "__main__":
    # method interpolating a exact function
    # my_f = lambda x:.001*x**3 - .1*x**2 - 4.3*x
    # demo_adapt(my_f, 3, 1e-2, 'monomial')
    # method interpolates a bessel function
    demo_adapt(f, 5, 1e-10, 'chebyshev')
    # see method interpolating a discontinuous function (must allow inaccuracies)
    demo_adapt(f1, 5, 1e-5, 'chebyshev', accurate=False)

