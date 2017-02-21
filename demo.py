"""
Demonstration of capabilities of the module

"""

import time
import numpy as np
import numpy.linalg as la
import scipy.special as spec
import matplotlib.pyplot as plt
from adaptive_interpolation import adapt
from adaptive_interpolation import generate
from adaptive_interpolation import approximator as app
from adaptive_interpolation import adaptive_interpolation as adapt_i


# bessel function for testing
def f(x):
    return spec.jn(0, x)
    # return x**2 - 10.*x**1 + 25.*x**0
    # return 0 + 0*x + (.5*(3*x**2 - 1))
    # it takes 40s to graph n =20 with 5e5 points


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
def my_plot(x, actual, approximation, abs_errors, allowed_error, ap_ranges):
    plt.figure()
    plt.title('Actual and Approximate values Graphed')
    plt.plot(x, actual, 'r')
    plt.plot(x, approximation, 'b')

    plt.figure()
    plt.yscale('log')
    plt.title('Absolute Error in Interpolated Values')
    plt.plot(x, abs_errors+1e-17, 'gs')
    plt.plot(x, 0*x + allowed_error, 'r')
    ranges = []
    for r in ap_ranges:
        try:
            ranges.append(r[0])
            ranges.append(r[1])
        except:
            pass
    print allowed_error, ranges
    plt.plot(ranges, 0*np.array(ranges) + allowed_error, 'bs')
    plt.show()


def demo_adapt(with_pyopencl=True):
    a, b, allowed_error = 0, 20, 1e-9
    my_approx = adapt_i.make_chebyshev_interpolant(a, b, f, 20, allowed_error)
    code = adapt_i.generate_code(my_approx)
    print(code)
    x = np.linspace(a, b, 1e2).astype(np.float64)
    if with_pyopencl:
        est = adapt_i.run_code(code, x, my_approx)
    else:
        est = my_approx.evaluate(x)
    true = f(x)
    my_plot(x, true, est, abs(true-est), allowed_error, my_approx.ranges)


def demo_adapt_variable(with_pyopencl=True):
    a, b, allowed_error = 0, 10, 1e-5
    my_approx = adapt_i.make_chebyshev_interpolant(a, b, f1, 5, allowed_error, True)
    code = adapt_i.generate_code(my_approx)
    print(code)
    x = np.linspace(a, b, 1e5).astype(np.float64)
    if with_pyopencl:
        est = adapt_i.run_code(code, x, my_approx)
    else:
        est = my_approx.evaluate(x)
    true = f1(x)
    my_plot(x, true, est, abs(true-est), allowed_error, my_approx.ranges)


# run the main program
if __name__ == "__main__":
    #Testing(0, 5, f, 30, 1e-14)
    demo_adapt()
    #demo_adapt_variable()
