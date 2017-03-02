"""
class that takes calculated approximation and
allows adaptive interpolant to be evaluated

A cleaner and more efficient rewrite
of simpleAdapt.py

Will also easily implement different node choices,
interpolant choices, etc.
"""
from __future__ import division

import numpy as np


# class to evaluate an adaptive interpolant that has different
# coefficients on different ranges
class Approximator(object):

    def __init__(self, adapt):
        # raw array data from adaptive interpolation method
        self.adapt = adapt
        self.heap = adapt.heap
        self.upper_bound = adapt.upper_bound
        self.lower_bound = adapt.lower_bound
        self.basis = adapt.basis
        self.code = 0
        self.max_order = adapt.max_order
        self.num_levels = int(np.log(len(self.heap))/np.log(2))
        self.midpoints, self.coeff = self.make_mid_coeff()
        self.used_coeff = self.heap[-2**(self.num_levels-1):]
        self.ranges, self.rel_errors = self.make_ranges_err()
        self.used_midpoints = self.midpoints[-2**(self.num_levels-1):]
        self.coeff_1d = self.make_vector_coeff()

    def make_vector_coeff(self):
        length = int(len(self.coeff)*(self.max_order + 1))
        coeff_1d = np.ones(length, dtype=np.float64)
        max_or = int(self.max_order+1)
        for i in range(length):
            coeff_1d[i] = self.coeff[i//max_or][i%max_or]
        return coeff_1d

    def make_mid_coeff(self):
        midpoints = [0]
        # initialize first element as coeff vector of 0 values
        # assume that all coeff vectors are numpy arrays in heap
        coeff = [0*self.heap[1][1]]
        # assume that the heap is properly allocated
        for i in range(1, len(self.heap)):
            midpoints.append(self.heap[i][0])
            coeff.append(self.heap[i][1])
        midpoints = np.array(midpoints, dtype=np.float64)
        return midpoints, coeff

    def make_ranges_err(self):
        ranges = []
        rel = []
        for r in self.used_coeff:
            try:
                ranges.append([r[3][0], r[3][1]])
                rel.append(r[4])
            except:
                pass
        return ranges, rel

    def get_index(self, x_val):
        # start at index 1
        index = 1
        for _ in range(self.num_levels):
            # get midpoint of the current interval
            mid = self.heap[index][0]
            # go left
            if x_val < mid:
                # if the next child does not exist return current element
                if 2*index >= len(self.heap):
                    return index
                # otherwise move to next level in tree
                else:
                    index = 2*index
            # otherwise go right
            else:
                if 2*index + 1 >= len(self.heap):
                    return index
                else:
                    index = 2*index + 1
        # reutrn the index of the element in the heap
        return index

    # assume that the x array being evaluated is increasing
    def evaluate(self, x):
        new_x = []
        for x0 in x:
            # get index of heap element for x0 and data from that element
            index = self.get_index(x0)
            coeff = self.heap[index][1]
            basis = self.heap[index][2]
            order = np.float64(len(coeff) - 1)
            # evaluate the given basis function
            xs = self.adapt.basis_function(x0, order, basis)
            # multiply the calculated basis by their coefficients
            # that are givent for the calculated array
            val = np.dot(np.array(coeff), xs)
            new_x.append(val)
        return np.array(new_x, dtype=np.float64)
