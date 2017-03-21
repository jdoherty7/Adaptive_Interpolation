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

try: import matplotlib.animation as animation
except: animate=False

try:
    with_pyopencl = True
    import pyopencl
except:
    with_pyopencl = False


# bessel function for testing
def f(x):
    return np.sin(1./x)
    return spec.jn(0, x)

def g(x):
    return (1. - np.exp(-(1.1)*(x-1)))**2

def morse_potential(x):
    return g(x)/g(.2)

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
               adapt_type="Trivial", accurate=True, animate=False, a=0, b=10):

    print("Creating Interpolant")
    my_approx = adapt_i.make_interpolant(a, b, function, order, 
                            allowed_error, basis, adapt_type, accurate)

    print("\nGenerated Code:\n")
    code = adapt_i.generate_code(my_approx)
    print(code)
    print("Evaluating Interpolant")
    x = np.linspace(a, b, 1e4, dtype=np.float64)
    if with_pyopencl: 
        est = adapt_i.run_code(x, my_approx)
    else: 
        est = my_approx.evaluate(x)
    print("Evaluating Function")
    true = function(x)
    print("Plotting")
    if animate:
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,2)
        ax2 = fig.add_subplot(1,2,1)
        ax1.set_ylim(-2, 2)
        ax2.set_yscale("log")
        ims = []
        # traverse levels 1 to end
        for i in range(int(my_approx.num_levels)):
            print("Plotting Level: ", i, '/', my_approx.num_levels-1)
            ims.append([])
            im0, = ax1.plot(x, function(x), 'r')
            rel, = ax2.plot(x, 0*x+allowed_error, 'r')
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.set_title("True Function vs. Approximation")
            ax2.set_xlabel("x")
            ax2.set_ylabel("Errors")
            ax2.set_title("Relative and Absolute Errors on Intervals")
            ims[i].append(im0)
            ims[i].append(rel)
            curr_level_size = 2**i
            for j in range(curr_level_size):
                k = curr_level_size + j
                if j==curr_level_size-1:
                    anim = True
                else:
                    if my_approx.heap[k][0] != my_approx.heap[k+1][0]:
                        anim = True
                    else:
                        anim = False
                # only animate if unique leaf
                if anim:
                    a0, b0 = my_approx.heap[k][3][0], my_approx.heap[k][3][1]
                    t = np.linspace(a0, b0, 1e3)
                    coeff = my_approx.heap[k][1]
                    y = []
                    for l in range(len(t)):
                        ys = my_approx.basis_function(t[l], len(coeff)-1, 
                                                my_approx.heap[k][2], a0, b0)
                        ys = np.dot(coeff, ys)
                        y.append(ys)
                    er = abs(np.array(y) - function(t))
                    #im1  = ax1.axvline(x=a0)
                    im2, = ax1.plot(t, y, 'b')
                    im3, = ax2.plot(t, er, 'g')
                    im4  = ax2.axvline(x=a0)
                    im5, = ax2.plot(my_approx.heap[k][0], my_approx.heap[k][4], 'bs')
                    #ims[i].append(im1)
                    ims[i].append(im2)
                    ims[i].append(im3)
                    ims[i].append(im4)
                    ims[i].append(im5)
            im6 = ax2.axvline(x=b0)
            ims[i].append(im6)
        ani = animation.ArtistAnimation(fig, ims, interval=1000)
        ani.save("adapt.mp4")
        plt.show()
    else:
        my_plot(x, true, est, abs(true-est), allowed_error, my_approx)


def main_demo():
    print("\nIn this demo three functions will be evaluated and")
    print("plotted, demonstrating some capabilities of this package.")
    print("This includes a special function, a highly oscillatory")
    print("function and a discontinuous function.")
    print("The code generated to evaluate each function will also be displayed.")
    # method interpolates a bessel function
    print("\n0th order Bessel Function")
    demo_adapt(f, 10, 1e-13, 'chebyshev', 'Remez')
    # method interpolating a exact function
    print("\nsin(1/x)")
    my_f = lambda x: np.sin(np.float64(1.)/x)
    demo_adapt(my_f, 20, 1e-10, 'chebyshev', a=.01, b=1)
    # variable order interpolation method
    print("\nA piecewise function")
    demo_adapt(f1, 6, 1e-4, 'chebyshev', 'Variable')


# run the main program
if __name__ == "__main__":
    #main_demo()
    demo_adapt(f, 20, 1e-4, 'chebyshev', adapt_type='remez', animate=True, a=.03, b=1 )

