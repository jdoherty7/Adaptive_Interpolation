"""
Script used to test the adaptive interpolation and
the evaluation of said interpolant
"""
import time
import generate_C
import numpy as np
import adapt2 as adapt
import numpy.linalg as la
import scipy.special as spec
import approximator as approx
import matplotlib.pyplot as plt



def f(x):
    return spec.jn(10, x)
    #it takes 40s to graph n =20 with 5e5 points


def my_plot(x, actual, approximation, abs_errors, rel_errors, err):
    plt.figure()
    plt.title('Actual and Approximate values Graphed')
    plt.plot(x, actual, 'r')
    plt.plot(x, approximation, 'b')

    plt.figure()
    plt.title('Absolute Error in Interpolated Values')
    plt.plot(x, abs_errors, 'gs')

    #plt.figure()
    #plt.title('Relative Error in Interpolated Values')
    #plt.plot(x, x*0+err, 'r')
    #plt.yscale('log')
    #plt.plot(x, rel_errors, 'bs')

    plt.show()
    
    
    

def Main(choice, order, int_c):
    #interpolant parameters
    a = 120         #lower bound of evaluation interval
    b = 140         #upper bound of interval
    err = 1e-3     #maximum error allowed in the approximation
    nt = choice    #node type used random and cheb are options, otherwise equispaced is used
    order = order     #order of the monomial interpolant to be used
    interp_choice = int_c #sine, legendre, or monomials
    
    
    start = time.time()
    print("Start adaptive interpolation")
    raw_interpolant_data = adapt.adaptive(f, a, b, err, nt, order, interp_choice)
    al_time = time.time() - start
    
    #the choice of interpolation is not currently implemented. monomials is default
    start = time.time()
    print("Building Approximator")
    my_approximation = approx.Approximator(raw_interpolant_data, interp_choice, order)
    setup_time = time.time() - start
    #print "Evaluating"
    #print
    #evaluate the interpolated approximation on values in x

    #print(generate_C.generate_srting(my_approximation))    
    
    x = np.linspace(a, b, 1e4)
    start = time.time()
    estimated_values = my_approximation.evaluate(x)
    eval_time = time.time() - start
    

    #calculate errors in the approximation and actual values
    start = time.time()
    actual_values = f(x)
    their_time = time.time() - start
    abs_errors = np.abs(actual_values - estimated_values)
    rel_error  = la.norm(abs_errors, np.inf)/la.norm(actual_values, np.inf)
    
    max_abs_error = np.max(abs_errors)
    avg_abs_error = np.sum(abs_errors)/(len(abs_errors))
    #max_rel_error = np.max(rel_errors)
    #avg_rel_error = np.sum(rel_errors)/(len(abs_errors))

    
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
    #print("Average relative error: ", avg_rel_error)
    print()
    
    my_plot(x, actual_values, estimated_values, abs_errors, rel_errors, err)
    
    return [max_abs_error, avg_abs_error, rel_error, eval_time]

#run the main program
if __name__ == "__main__":
    #for choice in ['chebyshev', 'random', 'equispaced']:
    for choice in ['equi']:
        for j in [5]:
            for interpolant in [ 'monomials']:
                #my_dict[choice][j][interpolant] = Main(choice , j, interpolant)
                Main(choice , j, interpolant)
