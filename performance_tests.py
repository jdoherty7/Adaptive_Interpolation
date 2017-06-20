"""
Script used to test the adaptive interpolation and
the evaluation of said interpolant

This is now faster than default bessel approximation!
"""
from __future__ import absolute_import

import os
import time
import numpy as np
import numpy.linalg as la
import scipy.special as spec
import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import adaptive_interpolation.adapt as adapt
import adaptive_interpolation.approximator as app
import adaptive_interpolation.generate as generate
import adaptive_interpolation.adaptive_interpolation as adapt_i



# bessel function for testing
def f(x, order=0):
    return spec.jn(order, x)


def f0(x, v):
    if v == 0:
        return f(x)
    elif v == 1:
        return spec.jn(10, x)
    elif v== 2:
        return spec.hankel1(0, x)
    elif v == 3:
        return spec.hankel1(10, x)
    elif v == 4:
        return spec.hankel2(0, x)
    elif v == 5:
        return spec.hankel2(10, x)
    else:
        return spec.airy(x)


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
    n = 10
    throw_out=40
    a, b = 0, 20
    sizes = 2**np.arange(1, 13)
    #sizes = np.linspace(1e2, 5e6, 5, dtype=np.int)
    tests = []
    orders = [9, 16]
    tests.append(0*np.zeros(sizes.shape))
 
    tests.append(0*np.zeros(sizes.shape))
    for j in range(len(orders)):
        tests.append(0*np.zeros(sizes.shape))
        approx = adapt_i.make_interpolant(a, b, f, orders[j], 1e-9, 'chebyshev')
        if True: # test interpolant is accurate
            y = np.linspace(a, b, 8*5e3)
            adapt_i.generate_code(approx, 8*5e3, 32)
            knl, q, xd, yd, treed = generate.build_code(y, approx)
            _, z = generate.run_single(approx)
            rel_err = la.norm(z-f(y),np.inf)/la.norm(f(y), np.inf)
            print("rel_error", orders[j], rel_err)
        for i in range(sizes.shape[0]):
            index = 0
            x = np.linspace(a, b, sizes[i])
            adapt_i.generate_code(approx, sizes[i], 1)
            knl, q, xd, yd, treed = generate.build_code(x, approx)
            for trial in range(n+throw_out):
                print("order: "+repr(j)+"/"+repr(len(orders))+"\ttrial:"+repr(trial+1)+"/"+repr(n+throw_out)+"\r")
                run_time, _ = generate.run_single(approx)
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
                            print("scipy", run_time, sizes[i], trial-throw_out)
                        tests[0][i] += run_time
            print()


    # average out each test
    for i in range(len(tests)):
        tests[i] /= float(n)

    fig = plt.figure()
    plt.title("Runtimes 1E-9 prec. bessel 0-20, {0} trials, vw=1".format(n))
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
    #plt.show()
    string = "data/t"+repr(time.time())+"n"+repr(n)+"vw1o"+repr(orders[0])+".png"
    fig.savefig(string)




def test_throughput(n, d, precision, size):
    if d != '32' and d != '64': return
    throw_out = 20
    a, b = 0, 20
    vws = [1, 2, 4, 8, 16, 32, 64]
    size = 2**14
    GB = size * float(d) / (8*2**20) # number of bits / a GB = # of GB
    orders = [9]

    tests = [[0 for __ in range(len(vws))] for _ in range(len(orders)+ 1)]

    for j in range(len(orders)):
        approx = adapt_i.make_interpolant(a, b, f, orders[j], precision, 'chebyshev', dtype=d)
        for v in range(len(vws)):
            # see how much time to process array
            adapt_i.generate_code(approx, size, vws[v])
            print()
            knl, q, treed = generate.build_code(approx)
            print(approx.code)
            for trial in range(n+throw_out):
                print("order: "+repr(j)+"/"+repr(len(orders))+"\ttrial:"+repr(trial+1)+"/"+repr(n+throw_out)+"\r")
                o = np.float32 if d == '32' else np.float64
                x = np.random.uniform(a, b, size=size).astype(o)
                run_time, _ = generate.run_single(x, approx)
                # run code multiple times before actually adding to tests
                if trial > throw_out:
                    tests[j+1][v] += GB/run_time
                    # only evaluate scipy's speed the first time
                    if j == 0:
                        start_time = time.time()
                        val = f(x, v)
                        run_time = time.time() - start_time
                        tests[0][v] += GB/run_time
    print()

    # average out each test
    #tests[0][0] /= float(n)
    for i in range(len(tests)):
        for j in range(len(vws)):
            tests[i][j] /= float(n)

    fig = plt.figure()
    plt.title("throughput {0} single, {1} trials, vw={2}".format(precision, n, vws[0]))
    plt.xlabel("Function Evaluated")
    plt.ylabel("Average Throughput (GB/s)")
    #plt.yscale("log")
    #plt.xscale("log")
    #plt.bar(0, tests[0][0], width=.5, align='center', color='r')
    i = 0
    z = np.linspace(-.2, .2, len(vws))
    colors = ['b', 'g', 'y', 'k', 'm', 'c']
    for v in range(len(vws)):
        plt.bar(i+z[v], tests[i][v], width=.3/len(vws), align='center', color=colors[i])
    xticks = ['scipy specials']
    for order in orders:
        z = np.linspace(-.2, .2, len(vws))
        for v in range(len(vws)):
            plt.bar(i+1+z[v], tests[i+1][v], width=.3/len(vws), align='center', color=colors[i])
        i+=1
        xticks.append("{0}th order approx".format(order))
    plt.xticks(range(len(orders)+1), xticks)
    #plt.show()
    string = "../data/00"+repr(d)+"t"+repr(time.time()%100)+"n"+repr(n)+"+vw"+repr(vws[0])+repr(vws[-1])+"o"+repr(orders[0])+repr(precision)+repr(size)+".png"
    fig.savefig(string)



# run the main program
if __name__ == "__main__":
    #test_speed()
    #test_throughput()

    p = 1e-6
    for d in ['32', '64']:
        for size in [2**10, 2**14]:
            test_throughput(25, d, p, size)


    #test_cheb_surf_speed()
    #test_exact_interpolants()
    #test_guaranteed_accuracy()
    #test_all_parallel_methods()

