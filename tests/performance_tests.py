"""
Script used to test the adaptive interpolation and
the evaluation of said interpolant

This is now faster than default bessel approximation!
"""
from future import __absoulte_import__
from nose.tools import *

import time
import timeit
import numpy as np
import numpy.linalg as la
import scipy.special as spec
import matplotlib.pyplot as plt
import adaptive_interpolation.adapt as adapt
import adaptive_interpolation.approximator as app
import adaptive_interpolation.generate as generate
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
def my_plot(x, actual, approximation, abs_errors):
    plt.figure()
    plt.title('Actual and Approximate values Graphed')
    plt.plot(x, actual, 'r')
    plt.plot(x, approximation, 'b')

    plt.figure()
    plt.yscale('log')
    plt.title('Absolute Error in Interpolated Values')
    plt.plot(x, abs_errors+1e-17, 'gs')

    plt.show()


# Given a specific Approximator class, this will test how the
# performance and accuracy varies when the code is varied from branching
# and vectorized to not branching and not vectorized
def test_parallel(approx):
    size = 1e7
    interval = approx.heap[1][3]
    x = np.linspace(interval[0], inverval[1], size, dtype=np.float64)
    nb_nv = adapt_i.generate_code(approx, 0, 0)
    nb_v  = adapt_i.generate_code(approx, 0, 1)
    b_nv  = adapt_i.generate_code(approx, 1, 0, size)
    b_v   = adapt_i.generate_code(approx, 1, 1, size)

    # time run_code functions and return times
    t00 = time.time()
    val_00 = run_code(nb_nv, x, approx=0, vectorized=False)
    t00 = time.time() - t00
    t01 = time.time()
    val_01 = run_code(nb_v,  x, approx,   vectorized=True)
    t01 = time.time() - t01
    t10 = time.time()
    val_10 = run_code(b_nv,  x, approx=0, vectorized=False)
    t10 = time.time() - t10
    t11 = time.time()
    val_11 = run_code(b_v,   x, approx,   vectorized=True)
    t11 = time.time() - t11

    # function values are independent of generative method
    assert la.norm(val00 - val01, np.inf) < 1e-15
    assert la.norm(val00 - val10, np.inf) < 1e-15
    assert la.norm(val00 - val11, np.inf) < 1e-15
    assert la.norm(val01 - val10, np.inf) < 1e-15
    assert la.norm(val01 - val11, np.inf) < 1e-15
    assert la.norm(val10 - val11, np.inf) < 1e-15

    print("nb_nv\tnb_v\tb_nv\tb_v")
    print(t00,'\t', t01, '\t', t10,'\t', t11)
    return [t00, t01, t10, t11]


def test_all_parallel_methods():
    a, b = 0, 10
    est1 = adapt_i.make_interpolant(a, b, f, 3, 1e-9, "monomial")
    est2 = adapt_i.make_interpolant(a, b, f, 3, 1e-9, "chebyshev")
    est3 = adapt_i.make_interpolant(a, b, f, 3, 1e-9, "legendre")

    test_parallel(est1)
    test_parallel(est2)
    test_parallel(est3)


def test_exact_interpolants():
    order1 = lambda x: 3*x + 7
    order4 = lambda x: 4.123*x**4 - 5.6*x**3 - x**2 + 4.5
    order6 = lambda x: x**6 - 3*x**5 - 2*x**4 + x - 3
    order8 = lambda x: x**8 - 42*x**7 + 7.5*x**5 - 4.1234*x**4  - 1.2*x**2

    a, b = -10, 10
    x = np.linspace(a, b, 100, dtype=np.float64)
    est1 = adapt_i.make_interpolant(a,b,order1,1,1e-9, "monomial").evaluate(x)
    est4 = adapt_i.make_interpolant(a,b,order4,4,1e-9, "monomial").evaluate(x)
    est6 = adapt_i.make_interpolant(a,b,order6,6,1e-9, "monomial").evaluate(x)
    est8 = adapt_i.make_interpolant(a,b,order8,8,1e-9, "monomial").evaluate(x)

    assert la.norm(est1-order1(x), np.inf)/la.norm(order1(x), np.inf) < 1e-15
    assert la.norm(est4-order4(x), np.inf)/la.norm(order4(x), np.inf) < 1e-15
    assert la.norm(est6-order6(x), np.inf)/la.norm(order6(x), np.inf) < 1e-15
    assert la.norm(est8-order8(x), np.inf)/la.norm(order8(x), np.inf) < 1e-15


# tests that the returned interpolant is below the given error
def test_guaranteed_accuracy():
    func1 = lambda x: np.sin(np.sin(x))
    func2 = lambda x: np.cos(np.sin(x))
    func3 = lambda x: np.sqrt(x)

    a, b = -10, 10
    x = np.linspace(a, b, 100, dtype=np.float64)

    est31 = adapt_i.make_interpolant(a,b,func1,10,1e-3, "monomial").evaluate(x)
    est32 = adapt_i.make_interpolant(a,b,func2,10,1e-3, "chebyshev").evaluate(x)
    est33 = adapt_i.make_interpolant(a,b,func3,10,1e-3, "legendre").evaluate(x)

    est61 = adapt_i.make_interpolant(a,b,func1,10,1e-6, "monomial").evaluate(x)
    est62 = adapt_i.make_interpolant(a,b,func2,10,1e-6, "chebyshev").evaluate(x)
    est63 = adapt_i.make_interpolant(a,b,func3,10,1e-6, "legendre").evaluate(x)

    est91 = adapt_i.make_interpolant(a,b,func1,10,1e-9, "monomial").evaluate(x)
    est92 = adapt_i.make_interpolant(a,b,func2,10,1e-9, "chebyshev").evaluate(x)
    est93 = adapt_i.make_interpolant(a,b,func3,10,1e-9, "legendre").evaluate(x)

    assert la.norm(est31-func1(x), np.inf)/la.norm(func1(x), np.inf) < 1e-3
    assert la.norm(est32-func2(x), np.inf)/la.norm(func2(x), np.inf) < 1e-3
    assert la.norm(est33-func3(x), np.inf)/la.norm(func3(x), np.inf) < 1e-3

    assert la.norm(est61-func1(x), np.inf)/la.norm(func1(x), np.inf) < 1e-6
    assert la.norm(est62-func2(x), np.inf)/la.norm(func2(x), np.inf) < 1e-6
    assert la.norm(est63-func3(x), np.inf)/la.norm(func3(x), np.inf) < 1e-6

    assert la.norm(est91-func1(x), np.inf)/la.norm(func1(x), np.inf) < 1e-9
    assert la.norm(est92-func2(x), np.inf)/la.norm(func2(x), np.inf) < 1e-9
    assert la.norm(est93-func3(x), np.inf)/la.norm(func3(x), np.inf) < 1e-9


# run the main program
if __name__ == "__main__":
    test_exact_interpolants()
    test_guaranteed_accuracy()
    test_all_parallel_methods()
