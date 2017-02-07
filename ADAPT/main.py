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


# bessel function for testing
def f(x):
    return spec.jn(3, x)
    #return x**2 - 10.*x**1 + 25.*x**0
    #return 0 + 0*x + (.5*(3*x**2 - 1))
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
def my_plot(x, actual, approximation, abs_errors, rel_errors, err):
    plt.figure()
    plt.title('Actual and Approximate values Graphed')
    plt.plot(x, actual, 'r')
    plt.plot(x, approximation, 'b')

    plt.figure()
    plt.yscale('log')
    plt.title('Absolute Error in Interpolated Values')
    plt.plot(x, abs_errors, 'gs')

    plt.show()


# Funtion to test the code base
# a is the lower bound and b is the upper bound while func is the function
# being approximated. Max order is the maximum allowed order of the
# interpolant while max_error is the maximum relative error allowed 
# the error is with respect to the infinity norm)
def Testing(a, b, func, max_order, max_error):
    # interpolant parameters
    # maximum error allowed in the approximation
    err = max_error
    # node type used random and cheb are options, otherwise equispaced is used
    nt = 'chebyshev'
    # order of the monomial interpolant to be used
    order = max_order
    # sine, chebyshev, legendre, or monomials
    interp_choice = 'monomials'

    start = time.time()
    print("Start adaptive interpolation")
    raw_interpolant_data = adapt.adaptive(func, a, b, err, nt, order, interp_choice)
    al_time = time.time() - start

    #the choice of interpolation is not currently implemented. monomials is default
    start = time.time()
    print("Building Approximator")
    my_approximation = app.Approximator(raw_interpolant_data, order)
    setup_time = time.time() - start

    # evaluate the interpolated approximation on values in x
    size = 1e3
    x = np.linspace(a, b, size).astype(np.float64)
    print("Evaluating the Function")
    start = time.time()
    #code = generate.generate_string(size, my_approximation)
    #print(code)
    #estimated_values = generate.run_c(x, code)
    estimated_values = my_approximation.evaluate(x)
    eval_time = time.time() - start

    # calculate errors in the approximation and actual values
    start = time.time()
    actual_values = func(x)
    their_time = time.time() - start
    abs_errors = np.abs(actual_values - estimated_values)
    rel_error = la.norm(abs_errors, np.inf)/la.norm(actual_values, np.inf)

    max_abs_error = np.max(abs_errors)
    avg_abs_error = np.sum(abs_errors)/(len(abs_errors))

    print()
    print(order, nt)
    print()
    print("-------------------TIMES-------------------------------")
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

    my_plot(x, actual_values, estimated_values, abs_errors, rel_error, err)

    return [max_abs_error, avg_abs_error, rel_error, eval_time]


def Test(a, b, func, max_order, max_error):
    # interpolant parameters
    # maximum error allowed in the approximation
    err = max_error
    # node type used random and cheb are options, otherwise equispaced is used
    nt = 'chebyshev'
    # order of the monomial interpolant to be used
    order = max_order
    # sine, chebyshev, legendre, or monomials
    interp_choice = 'chebyshev'

    heap, adapt_class = adapt.adaptive(func, a, b, err, nt, order, interp_choice)
    my_approximation = app.Approximator(adapt_class)

    # evaluate the interpolated approximation on values in x
    size = 1e4
    x = np.linspace(a, b, size).astype(np.float64)
    code = generate.generate_code_chebyshev(my_approximation)
    #code = generate.generate_vec(my_approximation)
    #code1 = generate.generate_string(size, my_approximation)
    start = time.time()
    print
    print code
    print
    print my_approximation.coeff
    
    estimated_values = generate.run_vector_c(x, my_approximation.midpoints, my_approximation.coeff, code)
    print "vec time: ", time.time() - start
    #start1 = time.time()
    #estimated_values = generate.run_c(x, code1)
    #print "non-vec time: ", time.time() - start1
    #estimated_values = my_approximation.evaluate(x)

    # calculate errors in the approximation and actual values
    s = time.time()
    actual_values = func(x)
    print time.time() - s
    abs_errors = np.abs(actual_values - estimated_values)
    rel_error = la.norm(abs_errors, np.inf)/la.norm(actual_values, np.inf)
    max_abs_error = np.max(abs_errors)
    avg_abs_error = np.sum(abs_errors)/(len(abs_errors))

    print("----------------ERRORS---------------------------------")
    print("Maximum absolute error: ", max_abs_error)
    print("Average absolute error: ", avg_abs_error)
    print("Maximum relative error: ", rel_error)
    my_plot(x, actual_values, estimated_values, abs_errors, rel_error, err)

def make_monomial_interpolant():
    pass
def make_chebyshev_interpolant()
    pass
    #make_legendre_interpolant()
    #generate_code(branching, vectorized)
    #save_code()
    #run_saved_code()
    #run_code(branching, vectorized)
    #generate_and_run_code()

# run the main program
if __name__ == "__main__":
    Test(1, 5, f, 9, 1e-5)
