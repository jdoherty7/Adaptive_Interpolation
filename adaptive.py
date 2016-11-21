# -*- coding: utf-8 -*-
"""
Class that finds the coefficients of an interpolation on
different intervals

Supercedes adapt2
"""
import numpy as np
import numpy.linalg as la


class Adaptive_Interpolation(object):

    # defining parameters of an adaptive method
    def __init__(self, f, error, interval, order):
        # function pass, must be vectorized
        self.function = f
        self.lower_bound = interval[0]
        self.upper_bound = interval[1]
        # max order allwed to create interpolation
        self.max_order = order
        # string specifying node choice
        self.node_choice = node_choice
        self.inter_array = []
        self.allowed_error = error

    # legendre polynomials evaluated in quick way
    def legendre(self, n, x):
        if n == 0:
            return np.array([1.])
        elif n == 1:
            return np.array([1., x])
        elif n > 1:
            L = [1., x]
            for i in range(2, n+1):
                first_term = (2*i-1)*x*L[i-1]
                second_term = (i-1)*L[i-2]
                L.append((first_term + second_term)*(1./n))
            return np.array(L)

    # function to evaluate the chebyshev polynomials
    def chebyshev(self, n, x):
        if n == 0:
            return np.array([1.])
        elif n == 1:
            return np.array([1., x])
        elif n > 1:
            C = [1., x]
            for i in range(2, n+1):
                first_term = 2*x*C[i-1]
                second_term = C[i-2]
                C.append(first_term + second_term)
            return np.array(C)

    # evaluate the given basis function returning an array of all values
    # of function up to given order.
    # if basis other than those specified is given, monomials is used
    def basis_function(self, x, order, basis):
        if (basis == 'legendre'):
            return self.legendre(order, x)
        elif (basis == 'chebyshev'):
            return self.chebyshev(order, x)
        else:
            return np.array([x**i for i in range(order+1)])

    # given a list of coefficients, evaluate what the interpolant's value
    # will be for the given x value(s). Assumes that x is an array
    # coeff is coefficients of a basis (string) of a given order (integer)
    def eval_coeff(self, coeff, x, basis, order):
        my_vals = []
        for x0 in x:
            xs = self.basis_function(x0, order, basis)
            val = np.dot(coeff, xs)
            my_vals.append(val)
        return np.array(my_vals)

    # get nodes for interpolation on the interval (a, b)
    def get_nodes(self, a, b, order, choice):
        node_number = order+1
        # choose nodes that are spaced like the chebyshev nodes
        if choice == 'chebyshev':
            k = np.array(range(1, int(node_number) + 1)[::-1])
            nodes = np.cos((2.*k - 2.)*np.pi/(2.*int(node_number-1)))
            # change range from -1 to 1 to a to b
            nodes = (b-a)*.5*(nodes + 1.) + a
            return nodes
        # choose nodes at random
        # beta function is used to prefer points near edges
        elif choice == 'random':
            nodes = (b-a)*np.random.beta(.5, .5, node_number) + a
            # make sure endpoints are properly set
            nodes[0], nodes[-1] = a, b
        # otherwise, create equispaced nodes
        else:
            nodes = np.linspace(a, b, node_number, endpoint=True)
        return nodes

    # find interpolated coefficients given a basis for
    # evaluation and nodes to evaluate the function at.
    def interpolate(self, nodes, basis):
        # the maximum order is 1 less than the number of nodes
        length = len(nodes)
        V = np.outer(np.ones(length), np.ones(length))
        # Build vandermonde matrix
        for i in range(length):
            V[i, :] = self.basis_function(nodes[i], length-1, basis)
        try:
            coeff = la.solve(V, self.function(nodes))
            return coeff
        except:
            # there is a singular matrix probably
            print(nodes)
            print(V)
            return [0]

    # find the error of given coefficients on interval a, b
    # with a given order an basis. Finds the relative error
    # using the infinity norm
    def Find_Error(self, coeff, a, b, basis, order):
        # check 100 points per unit. or 100 points total if interval
        # is smaller than 1. This should give an error, relatively stable
        # so long as dominant features are not smaller than this resolution
        eval_points = np.linspace(a, b, max(abs(b-a)*1e2, 1e2))
        actual = self.function(eval_points)
        approx = self.eval_coeff(coeff, eval_points, basis, order)
        # find maximum relative error in the given interval
        max_error = la.norm(actual - approx, np.inf)/la.norm(actual, np.inf)
        return max_error

    # adaptive method finding an interpolant for a function
    # this checks multiple bases and orders to make an interpolant
    def adapt(self, a, b):
        min_error = 1e14
        # check all the interpolant possibillities and orders to find the
        # best one that runs
        choice = 'legendre'
        for curr_order in range(self.max_order+1):
            for node_choice in ['random', 'chebyshev', 'random', 'mono']:
                nodes = self.get_nodes(a, b, curr_order, node_choice)
                curr_coeff = self.interpolate(nodes, choice)
                # if you get a singular matrix, break the for loop
                if curr_coeff[0] == 0:
                    break
                error = self.Find_Error(curr_coeff, a, b, choice, curr_order)
                print(error, node_choice, curr_order)
                if error < min_error:
                    coeff = curr_coeff
                    min_error = error
                    order = curr_order
                    basis = choice
        self.inter_array.append([coeff, [a, b], order, basis])
        if (min_error > self.allowed_error): # and (b-a) > 1e-3:
            # delete the parent array, which should be last added, because
            # it is no longer valid, since the interpolation has been refined
            del self.inter_array[-1]
            # adapt on the left subinterval and right subinterval
            self.adapt(a, (a+b)/2.)
            self.adapt((a+b)/2., b)
        else:
            print(a, b, min_error, basis, order, coeff)

    # Method to run the adaptive method initially
    def Adapt(self):
        self.adapt(self.lower_bound, self.upper_bound)
        return self.inter_array


# function that creates and then runs an adaptive method.
def adaptive(function, lower_bound, upper_bound, error, order):
    my_adapt = Adaptive_Interpolation(function, error, [lower_bound, upper_bound], order)
    array = my_adapt.Adapt()
    return array
