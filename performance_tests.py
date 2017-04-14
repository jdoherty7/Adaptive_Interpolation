"""
Script used to test the adaptive interpolation and
the evaluation of said interpolant

This is now faster than default bessel approximation!
"""
from __future__ import absolute_import
from nose.tools import *

import os
import time
import numpy as np
import numpy.linalg as la
import scipy.special as spec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#import adapt as adapt
#import approximator as app
#import generate as generate
#import adaptive_interpolation as adapt_i


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
    val_00 = adapt_i.run_code(x, approx=0, vectorized=False)
    t00 = time.time() - t00
    t01 = time.time()
    val_01 = adapt_i.run_code(x, approx,   vectorized=True)
    t01 = time.time() - t01
    t10 = time.time()
    val_10 = adapt_i.run_code(x, approx=0, vectorized=False)
    t10 = time.time() - t10
    t11 = time.time()
    val_11 = adapt_i.run_code(x, approx,   vectorized=True)
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
    x = np.linspace(a, b, 1e5, dtype=np.float64)
    est1 = adapt_i.make_interpolant(a,b,order1,1,1e-9, "monomial").evaluate(x)
    est4 = adapt_i.make_interpolant(a,b,order4,4,1e-9, "monomial").evaluate(x)
    est6 = adapt_i.make_interpolant(a,b,order6,6,1e-9, "monomial").evaluate(x)
    est8 = adapt_i.make_interpolant(a,b,order8,8,1e-9, "monomial").evaluate(x)
    plt.figure()
    plt.plot(x, np.sin(np.sin(x)), 'r')
    plt.plot(x, est1, 'b')
    plt.show()

    print(la.norm(est1-order1(x), np.inf)/la.norm(order1(x), np.inf))
    print(la.norm(est4-order4(x), np.inf)/la.norm(order4(x), np.inf))
    print(la.norm(est6-order6(x), np.inf)/la.norm(order6(x), np.inf))
    print(la.norm(est8-order8(x), np.inf)/la.norm(order8(x), np.inf))


    assert la.norm(est1-order1(x), np.inf)/la.norm(order1(x), np.inf) < 1e-15
    assert la.norm(est4-order4(x), np.inf)/la.norm(order4(x), np.inf) < 1e-15
    assert la.norm(est6-order6(x), np.inf)/la.norm(order6(x), np.inf) < 1e-15
    assert la.norm(est8-order8(x), np.inf)/la.norm(order8(x), np.inf) < 1e-15


# tests that the returned interpolant is below the given error
def test_guaranteed_accuracy():
    func1 = lambda x: np.sin(1./(x))
    func2 = lambda x: np.abs(x*np.sin(x))
    func3 = lambda x: np.sqrt(x)
    func4 = lambda x: np.abs(x*np.cos(x))

    a, b = 0.01, 10
    x = np.linspace(a, b, 1e5, dtype=np.float64)

    for func in [func4, func2, func3, func1]:
        for err in [1e-3, 1e-6, 1e-9]:
            for interpolant in ["monomial", "chebyshev", "legendre"]:
                est = adapt_i.make_interpolant(a,b,func,6,err, interpolant).old_evaluate(x)
                abs_err = la.norm(est-func(x), np.inf)
                rel_err = abs_err/la.norm(func(x), np.inf)
                print(interpolant, err, rel_err)
                plt.figure()
                plt.plot(x, func(x), 'r')
                plt.plot(x, est, 'b')
                plt.show()
                assert rel_err < err




def test_cheb_surf_speed():
    n = 4
    a, b = 0, 10
    orders = np.arange(8, 20, 2)
    sizes = np.arange(2, 8)
    #instantiate z
    z = []
    index = 0
    for _ in orders:
        z.append([])
        for _ in sizes:
            z[index].append(0)
        index+=1

    for _ in range(n):
        index_x=0
        for order in orders:
            index_y = 0
            approx = adapt_i.make_interpolant(a, b, f, order, 1e-5, "chebyshev")
            adapt_i.generate_code(approx, 0, 1)
            y = np.linspace(a, b, 1e3)
            print("rel_error", la.norm(approx.old_evaluate(y)-f(y),np.inf)/la.norm(f(y), np.inf))
            for i in sizes:
                x = np.linspace(a, b, 10**i)
                start_time = time.time()
                val = adapt_i.run_code(x, approx, vectorized=True)
                run_time = time.time() - start_time
                print(z)
                if _ > 1: #throw out first two trials
                    z[index_x][index_y] += run_time
                index_y+=1
            index_x+=1

    for x in range(len(z)):
        for y in range(len(z[x])):
            z[x][y] = z[x][y]/(n-2)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y = np.meshgrid(orders, sizes)
    ax.plot_surface(x, y, np.array(z))
    plt.show()


def test_speed():
    os.system("PYOPENCL_CTX=''")
    n, m = 1, 4
    a, b = 0, 50
    sizes = np.arange(5, 12)
    tests = []
    orders = [8, 16, 32]

    tests.append(0*np.ones(sizes.shape))
    for j in range(len(orders)):
        tests.append(0*np.ones(sizes.shape))
        for trial in range(n):
            approx = adapt_i.make_interpolant(a, b, f, orders[j], 1e-3, 'chebyshev')
            adapt_i.generate_code(approx, 0, 1)
            y = np.linspace(a, b, 5e4)
            z = adapt_i.run_code(y, approx, vectorized=True)
            plt.figure()
            plt.plot(y, f(y), 'r')
            plt.plot(y, z, 'b')
            plt.show()
            rel_err = la.norm(z-f(y),np.inf)/la.norm(f(y), np.inf)
            print("rel_error",orders[j],rel_err) #check its <1e-14
            for _ in range(m):
                index = 0
                for i in sizes:
                    x = np.linspace(a, b, 2**i)
                    start_time = time.time()
                    val = adapt_i.run_code(x, approx, vectorized=True)
                    run_time = time.time() - start_time
                    # run code twice before actually adding to tests
                    if trial > 0 or m >1:
                        tests[j][index] += run_time
                        start_time = time.time()
                        val = f(x)
                        run_time = time.time() - start_time
                        tests[0][index] += run_time
                        index+=1
    # average out each test
    for i in range(len(tests)):
        tests[i] /= (n*m - 2)


    plt.figure()
    plt.title("Runtimes of 10th order 10**-14 precision from 0-500, bessel")
    plt.xlabel("Size of evaluated array (2**x elements)")
    plt.ylabel("Time to evaluate (seconds)")
    a1, = plt.plot(sizes, tests[1], 'bs', label='16th order approx')
    sp, = plt.plot(sizes, tests[0], 'rs', label='scipy bessel')
    a2, = plt.plot(sizes, tests[2], 'gs', label='32nd order approx')
    a3, = plt.plot(sizes, tests[3], 'ys', label='64th order approx')
    plt.legend(handles = [a1, a2, a3, sp])
    plt.show()


# run the main program
if __name__ == "__main__":
    test_speed()
    #test_cheb_surf_speed()
    #test_exact_interpolants()
    #test_guaranteed_accuracy()
    #test_all_parallel_methods()
