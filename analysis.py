"""
Script used to test the adaptive interpolation and
the evaluation of said interpolant
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
        return spec.hankel2(0, x)
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



def test_throughput(n, d, precision, size):
    if d != '32' and d != '64': return
    throw_out = 20
    a, b = 0, 20
    vws = [1] # vector widths used
    GB = size * float(d) / (8*2**20) # number of bits / a GB in bits = # of GB
    orders = [1]

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
                        val = f0(x, v)
                        run_time = time.time() - start_time
                        tests[0][v] += GB/run_time
    print()

    # average out each test
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
    string = "../data/00" + repr(d) + "t" + repr(time.time()%100) + "n"
    string += repr(n) + "+vw" + repr(vws[0]) + repr(vws[-1]) + "o"
    string += repr(orders[0]) + repr(precision) + repr(size) + ".png"
    fig.savefig(string)



# run the main program
if __name__ == "__main__":
    test_throughput(1, '32', 1e0, 2**10)
    """
    p = 1e-6
    for d in ['32', '64']:
        for size in [2**10, 2**14]:
            test_throughput(25, d, p, size)
    """

