"""
Script used to test the adaptive interpolation and
the evaluation of said interpolant

This is now faster than default bessel approximation!
"""
import time
import generate
import numpy as np
import adapt
import numpy.linalg as la
import scipy.special as spec
import approximator as app
import matplotlib.pyplot as plt
import adaptive_interpolation as ai


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


# Funtion to test the code base
# a is the lower bound and b is the upper bound while func is the function
# being approximated. Max order is the maximum allowed order of the
# interpolant while max_error is the maximum relative error allowed 
# the error is with respect to the infinity norm)
def Testing(a, b, func, max_order, max_error):
    # this is set as so because double precision digits are used.    
    if max_error < 1e-15:
        max_error = 1e-15
    
    # interpolant parameters
    # maximum error allowed in the approximation
    err = max_error
    # node type used random and cheb are options, otherwise equispaced is used
    nt = 'chebyshev'
    # order of the monomial interpolant to be used
    order = max_order
    # sine, chebyshev, legendre, or monomials
    interp_choice = 'chebyshev'

    start = time.time()
    print("Start adaptive interpolation")
    my_adapt_class = adapt.adaptive(func, a, b, err, nt, order, interp_choice)
    al_time = time.time() - start

    #the choice of interpolation is not currently implemented. monomials is default
    start = time.time()
    print("Building Approximator")
    my_approximation = app.Approximator(my_adapt_class)
    setup_time = time.time() - start

    # evaluate the interpolated approximation on values in x
    size = 1e4
    x = np.linspace(a, b, size).astype(np.float64)
    print("Evaluating the Approximation")
    code = generate.gen_cheb_v(my_approximation)
    start = time.time()
    estimated_values = generate.run_c_v(x, my_approximation, code)
    #estimated_values = my_approximation.evaluate(x)
    eval_time = time.time() - start

    # calculate errors in the approximation and actual values
    start = time.time()
    print("Evaluating the Function")
    actual_values = func(x)
    their_time = time.time() - start
    abs_errors = np.abs(actual_values - estimated_values)
    rel_error = la.norm(abs_errors, np.inf)/la.norm(actual_values, np.inf)

    max_abs_error = np.max(abs_errors)
    avg_abs_error = np.sum(abs_errors)/(len(abs_errors))

    print()
    print("x size: ", size)
    print("max_order: ", order) 
    print("node choice: ", nt)
    print()
    print("-------------------TIMES-------------------------------")
    print("NOTE: The asymptotic values of the function evaluations")
    print("is what is important, which is not well represented here.")
    print("Time to run adaptive algorithm            : ", al_time)
    print("Time to construct approximation class     : ", setup_time)
    print("Time to evaluate the approximation  (MINE): ", eval_time)
    print("Time to evaluate precise function (THEIRS): ", their_time)
    print()
    print("----------------ERRORS---------------------------------")
    print("Maximum absolute error: ", max_abs_error)
    print("Average absolute error: ", avg_abs_error)
    print("Maximum relative error: ", rel_error)
    print()

    my_plot(x, actual_values, estimated_values, abs_errors)


# Given a specific Approximator class, this will test how the
# performance and accuracy varies when the code is varied from branching
# and vectorized to not branching and not vectorized
def test_parallel_methods(approx):
    pass

def ai_func_tests():
    a, b = 0, 20
    my_approximator = ai.make_chebyshev_interpolant(a, b, f, 20, 1e-9)
    code = ai.generate_code(my_approximator)
    print(code)
    x = np.linspace(a, b, 1e2).astype(np.float64)
    est = ai.run_code(code, x, my_approximator)
    #est = my_approximator.evaluate(x)
    true = f(x)
    my_plot(x, true, est, abs(true-est))


def dem_vary():
    a, b = 0, 10
    my_approximator = ai.make_chebyshev_interpolant(a, b, f1, 10, 1e-4, True)
    code = ai.generate_code(my_approximator)
    print(code)
    x = np.linspace(a, b, 1e5).astype(np.float64)
    est = ai.run_code(code, x, my_approximator)
    #est = my_approximator.evaluate(x)
    true = f1(x)
    my_plot(x, true, est, abs(true-est))


# run the main program
if __name__ == "__main__":
    #Testing(0, 5, f, 30, 1e-14)
    #ai_func_tests()
    dem_vary()
