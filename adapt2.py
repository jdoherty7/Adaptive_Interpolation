"""
New adaptive interpolation with better adaptive method

"""
import numpy as np
import numpy.linalg as la


class Adaptive_Interpolation(object):

    # defining parameters of an adaptive method
    def __init__(self, f, error, interval, order, node_choice, interpolant_choice):
        # function pass, must be vectorized
        self.function = f
        self.lower_bound = interval[0]
        self.upper_bound = interval[1]
        # max order allwed to create interpolation
        self.max_order = order
        # string specifying node choice
        self.node_choice = node_choice
        # string specifying basis choice
        self.basis = interpolant_choice
        self.inter_array = []
        self.allowed_error = error

    # function to evaluate order n Legendre polynomials. Is vectorized
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

    # gets chebyshev n nodes from a to b
    def get_cheb(self, a, b, n):
        if n == 1:
            return np.array([(a+b)/2.])
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
        return nodes

    # find interpolated coefficients given a basis for
    # evaluation and nodes to evaluate the function at.
    def interpolate(self, nodes, basis):
        length = len(nodes)
        V = np.outer(np.ones(length), np.ones(length))
        # Build vandermonde matrix
        for i in range(length):
            V[i, :] = self.basis_function(nodes[i], length-1, basis)
        try:
            print(nodes)
            print(V)
            print(self.function(nodes))
            coeff = la.solve(V, self.function(nodes))
            print(coeff)
            return coeff
        except:
            # there is a singular matrix probably
            #print(nodes)
            #print(V)
            return [0]

    # find the error of given coefficients on interval a, b
    # with a given order an basis. Finds the relative error
    # using the infinity norm
    def find_error(self, coeff, a, b, basis, order):
        # check 100 points per unit. This should give an error, relatively
        # stable so long as dominant features are not smaller than this resolution
        eval_points = np.linspace(a, b, max(abs(b-a)*1e2, 5e1)).astype(np.float64)
        actual = self.function(eval_points)
        approx = self.eval_coeff(coeff, eval_points, basis, order)
        # find maximum relative error in the given interval
        max_error = la.norm(actual - approx, np.inf)/la.norm(actual, np.inf)
        return max_error

    # adaptive method finding an interpolant for a function
    # this uses a specified order and basis function
    def adapt(self, a, b):
        # get nodes to evaluate interpolant with
        nodes = self.get_nodes(a, b, self.max_order)
        # get coefficients of interpolant defined on the nodes
        coeff = self.interpolate(nodes, self.basis)
        # append the coefficients and the range they are valid on to this array
        # also the basis function and order of in this range
        self.inter_array.append([coeff, [a, b], self.max_order, self.basis])
        # calculate the maximum relative error on the interval using these coefficients
        this_error = self.find_error(coeff, a, b, self.basis, self.max_order)
        # if error is larger than maximum allowed relative error then refine the interval
        if (this_error > self.allowed_error):
            # delete the parent array, which should be last added, because
            # it is no longer valid, since the interpolation has been refined
            del self.inter_array[-1]
            # adapt on the left subinterval and right subinterval
            self.adapt(a, (a+b)/2.)
            self.adapt((a+b)/2., b)

    # adaptive method finding an interpolant for a function
    # this checks multiple bases and orders to make an interpolant
    def order_adapt(self, a, b):
        min_error = 1e100
        # check all the interpolant possibillities and orders to find the
        # best one that runs
        coeff = [0]
        for curr_order in range(self.max_order+1):
            # only the monomial choice can be evaluated in the
            # for choice in ['chebyshev', 'legendre', 'sine', 'monomials']:
            for choice in ['monomials']:
                nodes = self.get_nodes(a, b, curr_order)
                curr_coeff = self.interpolate(nodes, choice)
                # if you get a singular matrix, break the for loop
                if curr_coeff[0] == 0:
                    break
                error = self.find_error(curr_coeff, a, b, choice, curr_order)
                #print(a, b, 'err', error, choice, curr_order)
                #print(nodes, curr_coeff)
                if error < min_error:
                    coeff = curr_coeff
                    min_error = error
                    order = curr_order
                    basis = choice
        #need to check that all these variables are actually assigned
        #print(a, b, "min error", min_error)
        if coeff[0] == 0:
            print(a, b)
            return
        self.inter_array.append([coeff, [a, b], order, basis])
        # if there is a discontinuity then b-a will be very small
        # but the error will still be quite large, the resolution
        # the second term combats that. 
        if (min_error > self.allowed_error):# or ((abs(b-a) < 1e-3)):
            # print(min_error, self.allowed_error, a, b, abs(b-a))
            # delete the parent array, which should be last added, because
            # it is no longer valid, since the interpolation has been refined
            del self.inter_array[-1]
            # adapt on the left subinterval and right subinterval
            self.order_adapt(a, (a+b)/2.)
            self.order_adapt((a+b)/2., b)
        else:
            print(a, b, min_error, basis, order, coeff)

    # Method to run the adaptive method initially
    def Adapt(self):
        self.order_adapt(self.lower_bound, self.upper_bound)
        return self.inter_array


# function that creates and then runs an adaptive method.
def adaptive(function, lower_bound, upper_bound, error, node_choice, order, interpolant_choice):
    my_adapt = Adaptive_Interpolation(function, error, [lower_bound, upper_bound], order, node_choice, interpolant_choice)
    array = my_adapt.Adapt()
    return array
