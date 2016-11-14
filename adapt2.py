"""
New adaptive interpolation with better adaptive method

"""

import numpy as np
import numpy.linalg as la


class Adaptive_Interpolation(object):

    def __init__(self, f, error, interval, order, node_choice, interpolant_choice):
        #defining parameters of an adaptive method
        self.function      = f #function pass, must be vectorized
        self.lower_bound   = interval[0]
        self.upper_bound   = interval[1]
        self.max_order     = order #integer
        self.node_choice   = node_choice #string
        self.basis         = interpolant_choice #string
        self.inter_array   = []
        self.allowed_error = error


    def Legendre(self, n, x):
        if n == 0:
            return 1.
        elif n == 1:
            return x
        elif n > 1:
            return ((2*n-1)*x*self.Legendre(n-1, x) - (n-1)*self.Legendre(n-2, x))*(1./n)

    #function to evaluate the chebyshev polynomials 
    def Chebyshev(self, n, x):
        if n == 0:
            return 1.
        elif n == 1:
            return x
        elif n > 1:
            return 2.*x*self.Chebyshev(n-1, x) - self.Chebyshev(n-2, x)
        
    #evaluate the given basis function for whatever order given
    def basis_function(self, x, order, basis):
        if (basis == 'sine'):
            if (order % 2) == 1:
                return np.sin(order*x)
            else:
                return np.cos(order*x)
        elif (basis == 'legendre'):
            return self.Legendre(order, x)
        elif (basis == 'chebyshev'):
            return self.Chebyshev(order, x)
        else:
            return x**order #monomials otherwise

    #given a list of coefficients, evaluate what the interpolant's value
    #will be for the given x value(s). Assumes that x is an array
    def eval_coeff(self, coeff, x, basis, order):
        my_vals = []
        for x0 in x:
            xs = np.array([self.basis_function(x0, i, basis) for i in range(order+1)])
            val = np.dot(coeff, xs)
            my_vals.append(val)
        return np.array(my_vals)
    
    #gets chebyshev nodes
    def get_cheb(self, a, b, n):
         k = np.array(range(1, int(n) + 1)[::-1])
         nodes = np.cos((2.*k - 2.)*np.pi/(2.*int(n-1)))
         #change range from -1 to 1 to a to b
         nodes = (b-a)*.5*(nodes + 1.) + a
         return nodes

    #get nodes for interpolation on the interval (a, b)
    def get_nodes(self, a, b, order):
        node_number = order+1
        #choose nodes that are spaced like the chebyshev nodes
        if self.node_choice == 'chebyshev':
            nodes = self.get_cheb(a, b, node_number)
        #choose nodes at random
        #beta function is used to prefer points near edges
        elif self.node_choice == 'random':
            nodes = (b-a)*np.random.beta(.5, .5, node_number) + a
            #make sure endpoints are properly set
            nodes[0], nodes[-1] = a, b
        #otherwise, create equispaced nodes
        else:
            nodes = np.linspace(a, b, node_number, endpoint=True)
        return nodes


    #find interpolated coefficients given a basis for evaluation and
    #nodes to evaluate the 
    def interpolate(self, nodes, basis):
        length = len(nodes)
        V = np.outer(np.ones(length), np.ones(length))
        #Build vandermonde matrix
        for i in range(length):
            for j in range(length):
                V[i, j] = self.basis_function(nodes[i], j, basis)
        coeff = la.solve(V, self.function(nodes))
        return coeff
    
        
    def Find_Error(self, coeff, a, b, basis, order):
        #check 100 points per unit. This should give an error, relatively
        #stable so long as dominant features are not smaller than this resolution
        eval_points = np.linspace(a, b, max(abs(b-a)*1e2, 5e1))
        actual = self.function(eval_points)
        approx = self.eval_coeff(coeff, eval_points, basis, order)
        #find maximum relative error in the given interval
        max_error = la.norm(actual - approx, np.inf)/la.norm(actual, np.inf)
        return max_error

    
    # adaptive method finding an interpolant for a function
    # this uses a specified order and basis function
    def adapt(self, a, b):
        #get nodes to evaluate interpolant with
        nodes = self.get_nodes(a, b, self.max_order)
        #get coefficients of interpolant defined on the nodes
        coeff = self.interpolate(nodes, self.basis)
        #append the coefficients and the range they are valid on to this array
        #also the basis function and order of in this range
        self.inter_array.append([coeff, [a,b], self.max_order, self.basis])
        #calculate the maximum relative error on the interval using these coefficients
        this_error = self.Find_Error(coeff, a, b, self.basis, self.max_order)
        #if error is larger than maximum allowed relative error then refine the interval
        if (this_error > self.allowed_error):
            #delete the parent array, which should be last added, because
            #it is no longer valid, since the interpolation has been refined
            del self.inter_array[-1]
            #adapt on the left subinterval and right subinterval
            self.adapt(a, (a+b)/2.)
            self.adapt((a+b)/2., b)


    # adaptive method finding an interpolant for a function
    #this checks multiple bases and orders to make an interpolant
    def order_adapt(self, a, b):
        min_error = 1e14
        #check all the interpolant possibillities and orders to find the
        #best one that runs
        for curr_order in range(self.max_order+1):
            #only the monomial choice can be evaluated in the 
            #for choice in ['chebyshev', 'legendre', 'sine', 'monomials']:
            for choice in ['monomials']:#['legendre']:
                nodes = self.get_nodes(a, b, curr_order)
                curr_coeff = self.interpolate(nodes, choice)
                error = self.Find_Error(curr_coeff, a, b, choice, curr_order)
                print(error, choice, curr_order)
                if error < min_error:
                    coeff = curr_coeff
                    min_error = error
                    order = curr_order
                    basis = choice
        self.inter_array.append([coeff, [a,b], order, basis])
        if (min_error > self.allowed_error):
            #delete the parent array, which should be last added, because
            #it is no longer valid, since the interpolation has been refined
            del self.inter_array[-1]
            #adapt on the left subinterval and right subinterval
            self.order_adapt(a, (a+b)/2.)
            self.order_adapt((a+b)/2., b)
        #else:
        #    print(a, b, min_error, basis, order, coeff)


    def Adapt(self):
        self.order_adapt(self.lower_bound, self.upper_bound)
        return self.inter_array


def adaptive(function, lower_bound, upper_bound, error, node_choice, order, interpolant_choice):
    my_adapt = Adaptive_Interpolation(function, error, [lower_bound, upper_bound], order, node_choice, interpolant_choice) 
    array = my_adapt.Adapt()
    return array