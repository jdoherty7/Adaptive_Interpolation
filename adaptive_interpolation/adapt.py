"""
New adaptive interpolation with better adaptive method
"""

from __future__ import division

import numpy as np
import numpy.linalg as la


class Interpolant(object):

    # defining parameters of an adaptive method
    def __init__(self, f, order, error, interpolant_choice, 
                 node_choice, guaranteed_accurate=True):
        if error <= 1e-16:
            string_err = "This package currently uses doubles thus an error"
            string_err+= "tolerance of less than 1e-16 is not possible."
            raise ValueError(string_err)
        my_bool = interpolant_choice != 'chebyshev'
        my_bool = my_bool and interpolant_choice != 'legendre'
        my_bool = my_bool and interpolant_choice != 'monomial'
        if my_bool:
            string_err = "{0} is not a valid interpolant.\n".format(interpolant_choice)
            string_err+= "legendre, chebyshev, and monomial are the choices."
            raise ValueError(string_err)
        # function pass, must be vectorized
        self.function = f
        self.lower_bound = 0
        self.upper_bound = 0
        # max number of recursion levels allowed for adaption
        # 34 reaches a spacing of 10**-15
        self.max_recur = 25
        # max order allwed to create interpolation
        self.max_order = order
        # string specifying node choice
        self.node_choice = node_choice
        # string specifying basis choice
        self.basis = interpolant_choice
        self.heap = [0, 0]
        self.allowed_error = error
        self.guaranteed_accurate = guaranteed_accurate

    # Equate every empty child with its parent so that the tree is full
    # // rounds down to obtain parent node
    def make_full_tree(self):
        for i in range(len(self.heap)):
            if self.heap[i] == 0:
                self.heap[i] = self.heap[i//2]

    # function to add data to the heap, if data being added is larger
    # than the current size of the heap then double its size (adds new level)
    def add_to_heap(self, data, index):
        if index < 1:
            raise ValueError("Error, heap index must be > 0")
        elif len(self.heap) <= index:
            for i in range(len(self.heap)):
                self.heap.append(0)
        self.heap[index] = data

    # function to evaluate Legendre polynomials of a number, x, up to order n
    def legendre(self, n, x):
        if n == 0:
            return np.array([1.], dtype=np.float64)
        elif n == 1:
            return np.array([1., x], dtype=np.float64)
        elif n > 1:
            L = [np.float64(1.), np.float64(x)]
            for i in range(2, int(n+1)):
                first_term = np.float64(2*i-1)*np.float64(x)*L[i-1]
                second_term = np.float64(i-1)*L[i-2]
                L.append((first_term + second_term)*(1./n))
            return np.array(L)

    # function to evaluate chebyshev polynomials of a value x up to order n
    def chebyshev(self, n, x):
        if n == 0:
            return np.array([1.], dtype=np.float64)
        elif n == 1:
            return np.array([1., x], dtype=np.float64)
        elif n > 1:
            C = [np.float64(1.), np.float64(x)]
            for i in range(2, int(n+1)):
                C.append(np.float64(2*x)*C[i-1] - C[i-2])
            return np.array(C)

    # transformation for othroganal functions, from [a, b] -> [-1, 1]
    def transform(self, x, a, b):
        scale = (x - a)/(b - a)
        return 2*scale - 1

    # given an order an a number, x. the polynomials of order 0 to n
    # are returned, evaluated for the given number.
    def basis_function(self, x, n, basis, a, b):
        if (basis == 'legendre'):
            return self.legendre(n, self.transform(x, a, b))
        elif (basis == 'chebyshev'):
            return self.chebyshev(n, self.transform(x, a, b))
        else:
            return np.array([x**i for i in range(int(order)+1)], dtype=np.float64)

    # given a list of coefficients, evaluate what the interpolant's value
    # will be for the given x value(s). Assumes that x is an array
    # coeff is coefficients of a basis (string) of a given order (integer)
    def eval_coeff(self, coeff, x, basis, order, a, b):
        my_vals = []
        for x0 in x:
            xs = self.basis_function(x0, order, basis, a, b)
            val = np.dot(coeff, xs)
            my_vals.append(val)
        return np.array(my_vals, dtype=np.float64)

    # gets n chebyshev nodes from a to b
    def get_cheb(self, a, b, n):
        if n == 1:
            return np.array([(a+b)/2.], dtype=np.float64)
        k = np.array(range(1, int(n) + 1)[::-1], dtype=np.float64)
        nodes = np.cos((2.*k - 2.)*np.pi/(2.*int(n-1)))
        # change range from -1 to 1 to a to b
        nodes = (b-a)*.5*(nodes + 1.) + a
        return nodes

    # get nodes for interpolation on the interval (a, b)
    def get_nodes(self, a, b, order):
        # create order + 1 number of nodes
        n = order+1
        # choose nodes that are spaced like the chebyshev nodes
        if self.node_choice == 'chebyshev':
            nodes = self.get_cheb(a, b, n)
        # otherwise, create equispaced nodes
        else:
            nodes = np.linspace(a, b, n, endpoint=True, dtype=np.float64)
        return nodes

    # find interpolated coefficients given a basis for
    # evaluation and nodes to evaluate the function at.
    def interpolate(self, nodes, basis, a, b):
        length = len(nodes)
        V = np.outer(np.ones(length), np.ones(length))
        # Build vandermonde matrix
        for i in range(length):
            V[i, :] = self.basis_function(nodes[i], length-1, basis, a, b)
        #print(a, b, "\t\t", la.cond(V))
        # try to solve for coefficients, if there is a singular matrix
        # or some other error then return [0] to indicate an error
        try: return la.solve(V, self.function(nodes))
        except: return [0]

    # find the error of given coefficients on interval a, b
    # with a given order an basis. Finds the relative error
    # using the infinity norm
    def find_error(self, coeff, a, b, order):
        # check 1000 points per unit. This should give an error,
        # relatively stable so long as dominant features are not
        # smaller than this resolution
        num_points = max(abs(b-a)*1000, self.max_order*100)
        eval_points = np.linspace(a, b, max(abs(b-a)*1e2, 1e2), dtype=np.float64)
        actual = self.function(eval_points)
        approx = self.eval_coeff(coeff, eval_points, self.basis, order, a, b)
        # find maximum relative error in the given interval
        rel_error = la.norm(actual - approx, np.inf)/la.norm(actual, np.inf)
        return rel_error

    # finds error using the max val as the max on the entire interval, not the current
    # below is the max number of points that can be evaluated exactly
    #(self.upper_bound - self.lower_bound)*(2**(self.max_recur+1))
    def find_error_new(self, coeff, a, b, order):
        n = min(5e3, int((b-a)/self.allowed_error)+1)
        lb, ub = self.lower_bound, self.upper_bound
        num_nodes = 100*(ub - lb)
        full_x = np.linspace(lb, ub, num_nodes, dtype=np.float64)
        x = np.linspace(a, b, n, dtype=np.float64)
        approx = self.eval_coeff(coeff, x, self.basis, order, a, b)
        actual = self.function(x)
        max_abs_err = la.norm(approx - actual, np.inf)
        max_val_full_int = la.norm(self.function(full_x), np.inf)
        return max_abs_err/max_val_full_int

    # adaptive method finding an interpolant for a function
    # this uses a specified order and basis function
    def adapt(self, a, b, index):
        # prevent from refining the interval too greatly
        # allow only 20 levels of refinement
        if (index >= 2**(self.max_recur+1)):
            string_err0 = "Recursed too far. Try changing the order of\n"
            string_err0+= "the interpolant used, raise the allowed error,\n"
            string_err0+= "or set accurate=False.\n"
            if self.guaranteed_accurate:
                raise ValueError(string_err0)
            else:
                return
        # get nodes to evaluate interpolant with
        nodes = self.get_nodes(a, b, self.max_order)
        # get coefficients of interpolant defined on the nodes
        # guaranteed to never give a singular matrix
        temp = self.interpolate(nodes, self.basis, a, b)
        if temp[0] != 0:
            coeff = temp
        else:
            string_err1 = "Singular matrix obtained on bounds [{0} {1}]\n".format(a, b)
            string_err1+= "If using monomials try using an orthogonal polynomial.\n"
            string_err1+= "Otherwise, try a different order interpolant, lower the\n"
            string_err1+= "allowed error, or set accurate=False\n"
            if self.guaranteed_accurate:
                raise ValueError(string_err1)
            else:
                return
        # calculate the maximum relative error on the interval
        # using these coefficients
        this_error = self.find_error_new(coeff, a, b, self.max_order)
        # append the coefficients and the range they are valid on to this
        # array also the basis function and order of in this range
        self.add_to_heap([(a+b)/2., coeff, self.basis, [a, b], this_error], index)
        # if error is larger than maximum allowed relative error
        # then refine the interval
        if (this_error > self.allowed_error):
            # adapt on the left subinterval then the right subinterval
            self.adapt(a, (a+b)/2., 2*index)
            self.adapt((a+b)/2., b, 2*index+1)

    # adaptive method finding an interpolant for a function
    # this checks multiple orders to make an interpolant
    def variable_order_adapt(self, a, b, index):
        # recursed too far, 15 levels down
        if (index >= 2**(self.max_recur+1)): 
            string_err0 = "Recursed too far. Try changing the order\n"
            string_err0+= "of the interpolant used, raise the allowed error,\n"
            string_err0+= "or set accurate=False."
            if self.guaranteed_accurate:
                raise ValueError(string_err0)
            else:
                return
        min_error = 1e100
        # check all the interpolant possibillities and
        # orders to find the best one that runs
        coeff = [0]
        # check orders 0 to max_order
        for curr_order in range(int(self.max_order)+1):
            nodes = self.get_nodes(a, b, curr_order)
            curr_coeff = self.interpolate(nodes, self.basis, a, b)
            # if you get a singular matrix, break the for loop
            if curr_coeff[0] == 0: break
            error = self.find_error_new(curr_coeff, a, b, curr_order)
            if error < min_error:
                coeff = curr_coeff
                min_error = error
        # need to check that all these variables are actually assigned
        if coeff[0] == 0: return
        # turn into max_order interpolant so it can run using
        # same generative code
        padded = np.zeros((int(self.max_order)+1,))
        padded[:coeff.shape[0]] = coeff
        self.add_to_heap([(a+b)/2., padded, self.basis, [a, b], min_error], index)
        if (min_error > self.allowed_error):
            self.variable_order_adapt(a, (a+b)/2., 2*index)
            self.variable_order_adapt((a+b)/2., b, 2*index + 1)


    ########################################################
    #                                                      #
    # Section Containing Functions for Remez interpolation #
    #                                                      #
    ########################################################


    # find interpolated coefficients given a basis for
    # evaluation and nodes to evaluate the function at.
    # n is order
    def solve_remez_system(self, nodes, n, a, b):
        length = nodes.shape[0]
        V = np.outer(np.ones(length), np.ones(length))
        for i in range(length):
            V[i, :-1] = self.basis_function(nodes[i], n, self.basis, a, b)
            V[i, -1] = (-1)**(i+1)
        try: return la.solve(V, function(nodes))
        except: return [0]

    # update node choices based on places with maximum error near
    # the current node choices, leave endpoints as is
    # if order 0 is used the nodes are not changed
    def update_nodes(self, nodes, coeff, n, a, b):
        if nodes.shape[0] > 2:
            err = lambda x: np.abs(self.eval_coeff(coeff, x, self.basis, n,
                                   a, b) - self.function(x))
            new_nodes = np.zeros(len(nodes))
            new_nodes[0] = nodes[0]
            new_nodes[-1] = nodes[-1]
            for i in range(1, len(nodes)-1):
                c, d = (new_nodes[i-1] + nodes[i])/2, (nodes[i] + nodes[i+1])/2
                x = np.linspace(c, d, 1e3)
                new_nodes[i] = x[np.argmax(err(x))]
            return new_nodes
        else:
            return nodes

    def check_eq_alt(self, array, error):
        tolerance = 1e-15
        the_sum = abs(np.sum(array))
        diff = abs(the_sum - abs(error))
        alternate = min(diff, the_sum)
        equal = abs(abs(array[0]) - error)
        if alternate <= tolerance and equal <=tolerance:
            print("Minimax Polynomial Found")
            return True
        else:
            print("Not Minimax Polynomial")
            return False

    def remez(self, a, b, n):
        remez_nodes = self.get_cheb(a, b, n+2)
        x = np.linspace(a, b, 1e3)
        #while (1):
        for _ in range(20):
            solution = self.solve_remez_system(remez_nodes, n, self.basis)
            if solution == [0]: return solution # singular matrix
            coeff = solution[:-1]
            error = abs(solution[-1])
            M = self.update_nodes(remez_nodes, coeff, n)
            err = lambda x: self.eval_coeff(coeff, x, self.basis, n,
                                            a, b) - self.function(x)
            if check_eq_alt(err(M), error): break
            if M.shape[0] == remez_nodes.shape[0]:
                remez_nodes = M
            else:
                raise ValueError("M not same size as X")
        return coeff


    # adaptive method utilizing the remez algorithm for interpolation
    def remez_adapt(self, a, b, index):
        if (index >= 2**(self.max_recur+1)):
            string_err0 = "Recursed too far. Try changing the order of\n"
            string_err0+= "the interpolant used, raise the allowed error,\n"
            string_err0+= "or set accurate=False.\n"
            if self.guaranteed_accurate:
                raise ValueError(string_err0)
            else:
                return
        # get coeff on interval utilizing the remez algorithm
        coeff = self.remez(a, b, self.max_order)
        if temp[0] != 0:
            coeff = temp
        elif self.guaranteed_accurate:
            string_err1 = "Singular matrix obtained on bounds [{0} {1}]\n".format(a, b)
            string_err1+= "If using monomials try using an orthogonal polynomial.\n"
            string_err1+= "Otherwise, try a different order interpolant, lower the\n"
            string_err1+= "allowed error, or set accurate=False\n"
            raise ValueError(string_err1)
        else:
            return
        this_error = self.find_error_new(coeff, a, b, self.max_order)
        self.add_to_heap([(a+b)/2., coeff, self.basis, [a, b], this_error], index)
        if (this_error > self.allowed_error):
            # adapt on the left subinterval then the right subinterval
            self.adapt(a, (a+b)/2., 2*index)
            self.adapt((a+b)/2., b, 2*index+1)


    # Method to run the adaptive method initially
    def run_adapt(self, lower_bound, upper_bound, adapt_type):
        if upper_bound <= lower_bound:
            raise Exception("Upper bound must be greater than lower bound.")
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if adapt_type == "Variable":
            self.variable_order_adapt(lower_bound, upper_bound, 1)
        elif adapt_type == "Remez":
            self.remez_adapt(lower_bound, upper_bound, 1)
        else:
            self.adapt(lower_bound, upper_bound, 1)
        self.make_full_tree()
        return self.heap


# function that creates and then runs an adaptive method.
def adaptive(lower_bound, upper_bound, function, order, error,
             interpolant_choice, node_choice,
             adapt_type="Trivial", guaranteed_accurate=True):
    my_adapt = Interpolant(function, np.float64(order),
                           np.float64(error), interpolant_choice,
                           node_choice, guaranteed_accurate)
    my_adapt.run_adapt(np.float64(lower_bound),
                       np.float64(upper_bound), variable)
    return my_adapt

