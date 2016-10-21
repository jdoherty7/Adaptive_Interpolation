"""
class that takes calculated approximation and
allows adaptive interpolant to be evaluated

A cleaner and more efficient rewrite
of simpleAdapt.py

Will also easily implement different node choices,
interpolant choices, etc.
"""
import numpy as np

#class to evaluate an adaptive interpolant that has different
#coefficients on different ranges
class Approximator(object):

    def __init__(self, array, interp_choice, order):
        #raw array data from adaptive interpolation method
        self.array = array
        #matrices of coefficients and the ranges they are valid on
        self.coeff, self.ranges = self.make_coeff()
        #the properties of the interpolant used, such as the
        #interpolant choice, and the order of the interpolant used
        self.basis= interp_choice
        self.order = order
        print()
        print(len(self.ranges))
        #for i in self.ranges:
            #print(i[1] - i[0])

    #function to evaluate the legendre polynomials
    def Legendre(self, n, x):
        if n == 0:
            return 1
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
            return 2*x*self.Chebyshev(n-1, x) - self.Chebyshev(n-2, x)
    
    #given a number/array and order the function evaluates it
    #based on the interpolant being used
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


    def make_coeff(self):
        coeff  = []
        ranges = []
        self.change_ranges()
        for i in range(len(self.array)):
            coeff.append(self.array[i][0])
            ranges.append(self.array[i][1])
        return self.organize(coeff, ranges)

        
    def change_ranges(self):
        #change lower bounds into respective ranges
        for i in range(len(self.array)):
            for j in range(i, len(self.array)):
                if self.array[i][1][0] == self.array[j][1][0]:
                    #check which one is a subinterval of the other
                    if self.array[i][1][1] > self.array[j][1][1]:
                        self.array[i][1][0] = self.array[j][1][1]
                    elif self.array[i][1][1] < self.array[j][1][1]:
                        self.array[j][1][0] = self.array[i][1][1]
        #for changing upper bounds
        for i in range(len(self.array)):
            for j in range(i, len(self.array)):
                if self.array[i][1][1] == self.array[j][1][1]:
                    #check which one is a subinterval of the other
                    if self.array[i][1][0] > self.array[j][1][0]:
                        self.array[j][1][1] = self.array[i][1][0]
                    elif self.array[i][1][0] < self.array[j][1][0]:
                        self.array[i][1][1] = self.array[j][1][0]

    #organize the coefficients and range arrays so they are in order
    def organize(self, coefficients, ranges):
        new_ranges   = []
        new_coeff    = []
        used_indices = []
        for i in range(len(ranges)):
            min_number = 1e9
            min_index  = 0
            for j in range(len(ranges)):
                if (ranges[j][0] < min_number) and (j not in used_indices):
                    min_number = ranges[j][0]
                    min_index  = j
            used_indices.append(min_index)
            new_ranges.append(ranges[min_index])
            new_coeff.append(coefficients[min_index])
        return new_coeff, new_ranges


    #get the index of the coefficients for the given number using
    #the known ranges of the coefficients.
    def get_index_of_range(self, number):
        for i in range(len(self.ranges)):
            #assume number is in the designated range
            if (self.ranges[i][0] <= number) and (number <= self.ranges[i][1]):
                return i
            else:
                #x is not in the interpolated interval
                return -1

    #assume that the x array being evaluated is increasing
    def evaluate(self, x):
        new_x = []
        #get the index of the range of the first number of the array
        index  = self.get_index_of_range(x[0])
        #x is not in interpolated interval
        if index == -1:
            print(x[0], 'is not in appropriate interval.')
            return 0
        for x0 in x:
            #if the number is not in the current range move the
            #range to the next range in the list
            #if self.in_current_range(x0, index) == False:
            if (self.ranges[index][1] < x0):
                index += 1
                #print index,' / ',len(self.ranges) 
            #make xs in the monomial series for evaluation
            #print len(self.coeff), index, x0, self.ranges[index][1], self.in_current_range(x0, index)
            xs = np.array([self.basis_function(x0, i) for i in range(self.order)])
            #multiply the calculated monomials by their coefficients
            #that are givent for the calculated array
            val = np.dot(np.array(self.coeff[index]), xs)
            new_x.append(val)
        return new_x