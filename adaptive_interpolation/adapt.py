"""
New adaptive interpolation with better adaptive method
"""

import numpy as np
import numpy.linalg as la


class Interpolant(object):

    # defining parameters of an adaptive method
    def __init__(self, f, error, order, node_choice, interpolant_choice):
        # function pass, must be vectorized
        self.function = f
        # max order allwed to create interpolation
        self.max_order = order
        # string specifying node choice
        self.node_choice = node_choice
        # string specifying basis choice
        self.basis = interpolant_choice
        self.heap = [0, 0]
        self.allowed_error = error

    def make_full_tree(self):
        for i in range(len(self.heap)):
            if self.heap[i] == 0:
                # if empty then equate to its parent, ints will round down.
                self.heap[i] = self.heap[i/2]

    # function to add data to the heap
    def add_to_heap(self, data, index):
        # if index is before the root
        if index < 1:
            print("Error, heap index must be > 0")
            return
        # if the index is beyond the size of the heap
        # then the heap size should be doubled
        elif len(self.heap) <= index:
            # double length of array if it is too small
            # this adds enough space for a new level in the tree
            for i in range(len(self.heap)):
                self.heap.append(0)
        # add to heap after possibly expanding the heap
        self.heap[index] = data

    # function to evaluate Legendre polynomials of a number, x, up to order n
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

    # function to evaluate chebyshev polynomials of a value x up to order n
    def chebyshev(self, n, x):
        if n == 0:
            return np.array([1.])
        elif n == 1:
            return np.array([1., x])
        elif n > 1:
            C = [1., x]
            for i in range(2, n+1):
                C.append(2*x*C[i-1] - C[i-2])
            return np.array(C)

    # evaluate the given basis function for whatever order given
    # if basis other than those specified is given, monomials is used
    # x can be array but this was built in the class for it to be a number
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

    # gets n chebyshev nodes from a to b
    def get_cheb(self, a, b, n):
        if n == 1:
            return np.array([(a+b)/2.]).astype(np.float64)
        k = np.array(range(1, int(n) + 1)[::-1])
        nodes = np.cos((2.*k - 2.)*np.pi/(2.*int(n-1)))
        # change range from -1 to 1 to a to b
        nodes = (b-a)*.5*(nodes + 1.) + a
        return nodes

    # get nodes for interpolation on the interval (a, b)
    def get_nodes(self, a, b, order):
        node_number = order+1
        # choose nodes that are spaced like the chebyshev nodes
        if self.node_choice == 'chebyshev':
            nodes = self.get_cheb(a, b, node_number)
        # choose nodes at random
        # beta function is used to prefer points near edges
        elif self.node_choice == 'random':
            nodes = (b-a)*np.random.beta(.5, .5, node_number) + a
            # make sure endpoints are properly set
            nodes[0], nodes[-1] = a, b
        # otherwise, create equispaced nodes
        else:
            nodes = np.linspace(a, b, node_number, endpoint=True)
        return nodes.astype(np.float64)

    # find interpolated coefficients given a basis for
    # evaluation and nodes to evaluate the function at.
    def interpolate(self, nodes, basis):
        length = len(nodes)
        V = np.outer(np.ones(length), np.ones(length)).astype(np.float64)
        # Build vandermonde matrix
        for i in range(length):
            V[i, :] = self.basis_function(nodes[i], length-1, basis)
        # try to solve for coefficients, if there is a singular matrix
        # or some other error then return [0] to indicate an error
        try: return la.solve(V, self.function(nodes))
        except: return [0]

    # find the error of given coefficients on interval a, b
    # with a given order an basis. Finds the relative error
    # using the infinity norm
    def find_error(self, coeff, a, b, order):
        # check 100 points per unit. This should give an error,
        # relatively stable so long as dominant features are not
        #  smaller than this resolution
        eval_points = np.linspace(a, b, max(abs(b-a)*1e2, 1e2))
        eval_points = eval_points.astype(np.float64)
        actual = self.function(eval_points)
        approx = self.eval_coeff(coeff, eval_points, self.basis, order)
        # find maximum relative error in the given interval
        rel_error = la.norm(actual - approx, np.inf)/la.norm(actual, np.inf)
        return rel_error

    # adaptive method finding an interpolant for a function
    # this uses a specified order and basis function
    def adapt(self, a, b, index):
        # prevent from refining the interval too greatly
        if (abs(b-a) < min(self.allowed_error, 1e-10)): return
        # get nodes to evaluate interpolant with
        nodes = self.get_nodes(a, b, self.max_order)
        # get coefficients of interpolant defined on the nodes
        # in new version replace with self.Remez(nodes)
        temp = self.interpolate(nodes, self.basis)
        if temp[0] != 0:
            coeff = temp
        else:
            print("Error assigning coeff", a, b)
            return
        # append the coefficients and the range they are valid on to this
        # array also the basis function and order of in this range
        self.add_to_heap([(a+b)/2., coeff, self.basis, [a, b]], index)
        # calculate the maximum relative error on the interval
        # using these coefficients
        this_error = self.find_error(coeff, a, b, self.max_order)
        # if error is larger than maximum allowed relative error
        # then refine the interval
        if (this_error > self.allowed_error):
            # adapt on the left subinterval then the right subinterval
            self.adapt(a, (a+b)/2., 2*index)
            self.adapt((a+b)/2., b, 2*index+1)

    # adaptive method finding an interpolant for a function
    # this checks multiple orders to make an interpolant
    def variable_order_adapt(self, a, b, index):
        # recursed too far
        if (abs(b-a) < self.allowed_error): return
        min_error = 1e100
        # check all the interpolant possibillities and orders to
        # find the best one that runs
        coeff = [0]
        # check orders 0 to max_order
        for curr_order in range(self.max_order+1):
            # only the monomial choice can be evaluated in the
            nodes = self.get_nodes(a, b, curr_order)
            curr_coeff = self.interpolate(nodes, self.basis)
            # if you get a singular matrix, break the for loop
            if curr_coeff[0] == 0: break
            error = self.find_error(curr_coeff, a, b, curr_order)
            if error < min_error:
                coeff = curr_coeff
                min_error = error
        # need to check that all these variables are actually assigned
        if coeff[0] == 0: return
        # turn into max_order interpolant so it can run using
        # same generative code
        padded = np.zeros((self.max_order+1,))
        padded[:coeff.shape[0]] = coeff
        self.add_to_heap([(a+b)/2., padded, self.basis, [a, b]], index)
        # if there is a discontinuity then b-a will be very small
        # but the error will still be quite large, the resolution
        # the second term combats that.
        if (min_error > self.allowed_error):
            # adapt on the left subinterval and right subinterval
            self.variable_order_adapt(a, (a+b)/2., 2*index)
            self.variable_order_adapt((a+b)/2., b, 2*index + 1)

    # Method to run the adaptive method initially
    def run_adapt(self, lower_bound, upper_bound, variable_order=False):
        if variable_order:
            self.variable_order_adapt(lower_bound, upper_bound, 1)
        else:
            self.adapt(lower_bound, upper_bound, 1)
        self.make_full_tree()
        return self.heap


# function that creates and then runs an adaptive method.
def adaptive(function, lower_bound, upper_bound, error, node_choice,
             order, interpolant_choice, variable=False):
    my_adapt = Interpolant(function, error, order, node_choice,
                           interpolant_choice)
    my_adapt.run_adapt(lower_bound, upper_bound, variable)
    return my_adapt
