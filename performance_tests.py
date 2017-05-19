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
    nb_nv = adapt_i.generate_code(approx)
    nb_v  = adapt_i.generate_code(approx)
    b_nv  = adapt_i.generate_code(approx)
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
                est = adapt_i.make_interpolant(a,b,func,6,err, interpolant).evaluate(x)
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
            adapt_i.generate_code(approx, 0, 1)
            y = np.linspace(a, b, 1e3)
            print("rel_error", la.norm(approx.evaluate(y)-f(y),np.inf)/la.norm(f(y), np.inf))
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
    n = 20
    throw_out=40
    a, b = 0, 20
    sizes = 2**np.arange(3, 19)
    #sizes = np.linspace(1e2, 5e6, 5, dtype=np.int)
    tests = []
    orders = [5, 9]
    tests.append(0*np.zeros(sizes.shape))
 
    tests.append(0*np.zeros(sizes.shape))
    for j in range(len(orders)):
        tests.append(0*np.zeros(sizes.shape))
        approx = adapt_i.make_interpolant(a, b, f, orders[j], 1e-9, 'chebyshev')
        y = np.linspace(a, b, 8*5e0)
        adapt_i.generate_code(approx, 8*5e0, 1)
        knl, q, xd, yd, treed = generate.build_code(y, approx)
        _, z = generate.run_single(approx)#knl, q, xd, yd, treed)
        if False:
            print(approx.code)
            plt.figure()
            plt.plot(y, f(y), 'r')
            plt.plot(y, z, 'b')
            plt.figure()
            plt.yscale("log")
            plt.plot(y, abs(z-f(y)), 'g')
            plt.show()
        rel_err = la.norm(z-f(y),np.inf)/la.norm(f(y), np.inf)
        print("rel_error",orders[j],rel_err) #check its <1e-14
        for i in range(sizes.shape[0]):
            index = 0
            x = np.linspace(a, b, sizes[i])
            adapt_i.generate_code(approx, sizes[i], 1)
            knl, q, xd, yd, treed = generate.build_code(x, approx)
            for trial in range(n+throw_out):
                print(i+1, "/", sizes.shape[0], "\ttrial:", trial+1, "/", n+throw_out, end="\r")
                #knl, q, xd, yd, treed = generate.build_test(x, approx)
                run_time, _ = generate.run_single(approx)#knl, q, xd, yd, treed)
                # run code multiple times before actually adding to tests
                if trial >= throw_out:
                    tests[j+1][i] += run_time
                    if j == 0:
                        if trial-throw_out > 17:
                            print("mine",run_time)
                        start_time = time.time()
                        val = f(x)
                        run_time = time.time() - start_time
                        #print(la.norm(_-val,np.inf)/la.norm(val, np.inf))
                        if trial - throw_out > 17:
                            print("scipy",run_time, sizes[i], trial-throw_out)
                        tests[0][i] += run_time
            print()


    # average out each test
    for i in range(len(tests)):
        tests[i] /= float(n)

    plt.figure()
    plt.title("Runtimes 10**-6 precision from 0-20, bessel, {0} trials, vector width=4".format(n))
    plt.xlabel("Size of evaluated array")
    plt.ylabel("Time to evaluate (seconds)")
    #plt.yscale("log")
    #plt.xscale("log")
    sp, = plt.plot(sizes, tests[0], 'r', label='scipy bessel')
    i = 0
    hand=[sp]
    colors = ['b', 'g', 'y', 'k', 'm', 'c']
    for order in orders:
        a1, = plt.plot(sizes, tests[i+1], colors[i],label="{0}th order approx".format(order))
        i+=1
        hand.append(a1)
    plt.legend(handles =hand)
    plt.show()



def test_throughput():
    n = 10
    throw_out = 40
    a, b = 0, 20
    precision = 1e-4
    vws = [1, 2, 4, 8, 16] # vector widths
    size = 2**10
    x = np.linspace(a, b, size, dtype=np.float64)
    GB = size * 16/(8*2**20) # size times 16 bytes for each float64/GB
    orders = [9]

    tests = [[0 for __ in range(len(vws))] for _ in range(len(orders)+ 1)]

    for j in range(len(orders)):
        approx = adapt_i.make_interpolant(a, b, f, orders[j], precision, 'chebyshev')
        # test that the error is below the desired error and function is right
        if False:
            y = np.linspace(a, b, 1000)
            adapt_i.generate_code(approx, 1000, vws[0])
            knl, q, xd, yd, treed = generate.build_code(y, approx)
            print(approx.code)
            _, z = generate.run_single(approx)#knl, q, xd, yd, treed, vws[0])
            rel_err = la.norm(z-f(y),np.inf)/la.norm(f(y), np.inf)
            print("rel_error",orders[j],rel_err) #check its <1e-14

        for v in range(len(vws)):
            # see how much time to process array
            adapt_i.generate_code(approx, size, vws[v])
            print()
            #print(approx.code)
            #print(approx.size, approx.vector_width)
            knl, q, xd, yd, treed = generate.build_code(x, approx)
            for trial in range(n+throw_out):
                print("order: ",j,"/",len(orders),"\ttrial:",trial+1,"/",n+throw_out,end="\r")
                run_time, _ = generate.run_single(approx)#knl, q, xd, yd, treed, vws[v])
                # run code multiple times before actually adding to tests
                if trial >= throw_out:
                    tests[j+1][v] += run_time
                    # only evaluate scipy's speed the first time
                    if v == 0 and j == 0:
                        start_time = time.time()
                        val = f(x)
                        run_time = time.time() - start_time
                        tests[0][0] += run_time
    print()

    # average out each test
    tests[0][0] /= float(n)
    for i in range(1, len(tests)):
        for j in range(len(vws)):
            tests[i][j] /= float(n)

    plt.figure()
    plt.title("avg runtime {0} precision from 0-20, bessel, {1} trials, vector width={2}-{3}".format(precision, n, vws[0], vws[-1]))
    plt.xlabel("Function Evaluated")
    plt.ylabel("Average Throughput (GB/s)")
    #plt.yscale("log")
    #plt.xscale("log")
    plt.bar(0, tests[0][0], width=.5, align='center', color='r')
    i = 0
    xticks = ['scipy bessel']
    colors = ['b', 'g', 'y', 'k', 'm', 'c']
    for order in orders:
        z = np.linspace(-.2, .2, len(vws))
        for v in range(len(vws)):
            plt.bar(i+1+z[v], tests[i+1][v], width=.3/len(vws), align='center', color=colors[i])
        i+=1
        xticks.append("{0}th order approx".format(order))
    plt.xticks(range(len(orders)+1), xticks)
    plt.show()


# run the main program
if __name__ == "__main__":
    test_speed()
    #test_throughput()
    #test_cheb_surf_speed()
    #test_exact_interpolants()
    #test_guaranteed_accuracy()
    #test_all_parallel_methods()
