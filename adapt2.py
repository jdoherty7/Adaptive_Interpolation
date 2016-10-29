"""
New adaptive interpolation with better adaptive method

"""

import numpy as np
import numpy.linalg as la


class Adaptive_Interpolation(object):

    def __init__(self, f, error, interval, order, node_choice, interpolant_choice):
        #defining parameters of an adaptive method
        self.function      = f #function pass
        self.lower_bound   = interval[0]
        self.upper_bound   = interval[1]
        self.order         = order #integer
        self.node_choice   = node_choice #string
        self.basis         = interpolant_choice #string
        self.inter_array   = []
        self.allowed_error = error


    def Legendre(self, n, x):
        if n > 1:
            return ((2*n-1)*x*self.Legendre(n-1, x) - (n-1)*self.Legendre(n-2, x))*(1./n)
        if n == 0:
            return 1
        if n == 1:
            return x

    #function to evaluate the chebyshev polynomials 
    def Chebyshev(self, n, x):
        if n == 0:
            return 1.
        elif n == 1:
            return x
        elif n > 1:
            return 2*x*self.Chebyshev(n-1, x) - self.Chebyshev(n-2, x)
        
    #evaluate the given basis function for whatever order given
    def basis_function(self, x, order):
        if (self.basis == 'sine'):
            if (order % 2) == 1:
                return np.sin(order*x)
            else:
                return np.cos(order*x)
        elif (self.basis == 'legendre'):
            return self.Legendre(order, x)
        elif (self.basis == 'chebyshev'):
            return self.Chebyshev(order, x)
        else:
            return x**order #monomials otherwise


    #given a list of coefficients, evaluate what the interpolant's value
    #will for the given x value(s)
    def eval_coeff(self, coeff, x):
        my_vals = []
        if type(x) != int:
            #evaluating for an array
            for x0 in x:
                xs = np.array([self.basis_function(x0, i) for i in range(self.order)])
                val = np.dot(coeff, xs)
                #print coeff
                #print xs
                #print 'val', val
                my_vals.append(val)
        else:
            xs = np.array([self.basis_function(x, i) for i in range(self.order)])
            val = np.dot(coeff, xs)
            my_vals.append(val)
        return np.array(my_vals)
    
    #gets chebyshev nodes
    def get_cheb(self, a, b, n):
         k = np.array(range(1, int(n) + 1)[::-1])
         nodes = np.cos((2.*k - 1.)*np.pi/(2*int(n)))
         #change range from -1 to 1 to a to b
         nodes = (b-a)*.5*(nodes + 1.) + a
         nodes[0], nodes[-1] = a, b
         return nodes

    #get nodes for interpolation on the interval (a, b)
    def get_nodes(self, a, b):
        #choose nodes that are spaced like the chebyshev nodes
        if self.node_choice == 'chebyshev':
            nodes = self.get_cheb(a, b, self.order)
        #choose nodes at random
        #beta function is used to prefer points near edges
        elif self.node_choice == 'random':
            nodes = (b-a)*np.random.beta(.5, .5, self.order) + a
        #otherwise, create equispaced nodes
        else:
            nodes = np.linspace(a, b, self.order, endpoint=True)
        #make sure endpoints are properly set
        nodes[0], nodes[-1] = a, b
        return nodes



    def interpolate(self, nodes):
        V = np.outer(np.ones(self.order), np.ones(self.order))
        #Build vandermonde matrix
        for i in range(self.order):
            for j in range(self.order):
                V[i, j] = self.basis_function(nodes[i], j)
        coeff = la.solve(V, self.function(nodes))
        return coeff
    
        
    def Find_Error(self, coeff, a, b):
        #the number of points used to check the error influence what the
        #actual error will end up being more than the parameter.
        #it also greatly affects run time
        eval_points = np.linspace(a, b, abs(b-a)*1e1/self.allowed_error)
        actual = self.function(eval_points)
        approx = self.eval_coeff(coeff, eval_points)
        #find maximum relative error in the given interval
        #this makes errors work... not la.norm though...
        #changing eval points affects it too, more is better
        #max_error = max(abs((actual - approx)/actual))
        #max_error = np.max(np.abs((actual - approx)/np.abs(actual)))
        #should error be defined this way, so that you get one value on the
        #entire interval, or like the way above
        max_error = la.norm(actual - approx, np.inf)/la.norm(actual, np.inf)
        return max_error

    
    # adaptive method finding an interpolant for a function
    def adapt(self, a, b):
        #get nodes to evaluate interpolant with
        nodes = self.get_nodes(a, b)
        #print(nodes)
        #get coefficients of interpolant defined on the nodes
        coeff = self.interpolate(nodes)
        #append the coefficients and the range they are valid on to this array
        self.inter_array.append([coeff, [a,b]])
        #recursisve call both because I guess I cant just call one then another later
        #for index in range(len(errors)):
        #calculate the maximum relative error on the interval using these coefficients
        this_error = self.Find_Error(coeff, a, b)
        #print(this_error)
        #if error is larger than maximum allowed relative error then refine the interval
        if (this_error > self.allowed_error):
            #delete the parent array, which should be last added, because
            #it is no longer valid, since the interpolation has been refined
            del self.inter_array[-1]
            #adapt on the left subinterval and right subinterval
            self.adapt(a, (a+b)/2.)
            self.adapt((a+b)/2., b)



    def Adapt(self):
        self.adapt(self.lower_bound, self.upper_bound)
        return self.inter_array


def adaptive(function, lower_bound, upper_bound, error, node_choice, order, interpolant_choice):
    my_adapt = Adaptive_Interpolation(function, error, [lower_bound, upper_bound], order, node_choice, interpolant_choice) 
    array = my_adapt.Adapt()
    return array