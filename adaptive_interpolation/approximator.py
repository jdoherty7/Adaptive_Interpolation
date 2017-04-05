"""
class that takes calculated approximation and
allows adaptive interpolant to be evaluated

A cleaner and more efficient rewrite
of simpleAdapt.py

Will also easily implement different node choices,
interpolant choices, etc.
"""
from __future__ import division
from __future__ import absolute_import

import numpy as np


# class to evaluate an adaptive interpolant that has different
# coefficients on different ranges
class Approximator(object):

    def __init__(self, my_adapt=None):
        if my_adapt != None:
            # raw array data from adaptive interpolation method
            self.basis_function = my_adapt.basis_function
            self.heap = my_adapt.heap
            self.upper_bound = my_adapt.upper_bound
            self.lower_bound = my_adapt.lower_bound
            self.basis = my_adapt.basis
            self.code = 0
            self.max_order = my_adapt.max_order
            self.num_levels = int(np.log(len(self.heap))/np.log(2))
            self.all_ranges, self.midpoints, self.coeff = self.make_mid_coeff()
            self.a_ranges0, self.a_ranges1 = self.make_run_ranges()
            self.used_coeff = self.coeff[-2**(self.num_levels-1):]
            self.a_ranges0 = self.a_ranges0[-2**(self.num_levels-1):] 
            self.a_ranges1 = self.a_ranges1[-2**(self.num_levels-1):]
            self.ranges, self.rel_errors = self.make_ranges_err()
            self.used_midpoints = self.midpoints[-2**(self.num_levels-1):]
            self.coeff_1d = self.make_vector_coeff()
            self.new_indices, self.run_coeff, self.ranges0, self.ranges1 = self.make_new_indices()
            self.run_vector= 0 # 1d vector with all necessary variables

        print(self.run_coeff)
        print(self.midpoints)
        print(self.ranges0)
        print(self.ranges1)
        print(self.new_indices)


    # map from index -> coeff_1d
    def make_new_indices(self):
        new_array = [0] # have first index in there already, 0
        run_coeff = [self.used_coeff[0]]
        range0 = [self.a_ranges0[0]]
        range1 = [self.a_ranges1[0]]
        index = 0
        last_coeff = self.used_coeff[0]
        for i in range(1, len(self.used_coeff)):
            if (not np.array_equal(last_coeff, self.used_coeff[i])):
                index+=1;
                run_coeff.append(self.used_coeff[i])
                range0.append(self.a_ranges0[i])
                range1.append(self.a_ranges1[i])
            new_array.append(index)
            last_coeff = self.used_coeff[i]
        return np.array(new_array), np.array(run_coeff), np.array(range0), np.array(range1)

    def make_vector_coeff(self):
        length = int(len(self.used_coeff)*(self.max_order + 1))
        coeff_1d = np.ones(length, dtype=np.float64)
        max_or = int(self.max_order+1)
        for i in range(length):
            coeff_1d[i] = self.used_coeff[i//max_or][i%max_or]
        return coeff_1d

    def make_mid_coeff(self):
        midpoints = [0]
        # initialize first element as coeff vector of 0 values
        # assume that all coeff vectors are numpy arrays in heap
        coeff = [0*self.heap[1][1]]
        all_range = [[0,0]]
        # assume that the heap is properly allocated
        for i in range(1, len(self.heap)):
            midpoints.append(self.heap[i][0])
            coeff.append(self.heap[i][1])
            all_range.append(np.array(self.heap[i][3], dtype=np.float64))
        midpoints = np.array(midpoints, dtype=np.float64)
        all_range = np.array(all_range, dtype=np.float64)
        return all_range, midpoints, coeff

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

    # make range arrays to pass to run_c_v
    def make_run_ranges(self):
        lower, upper = [], []
        for r in self.all_ranges:
            lower.append(r[0])
            upper.append(r[1])
        lower = np.array(lower, dtype=np.float64)
        upper = np.array(upper, dtype=np.float64)
        return lower, upper

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
    def old_evaluate(self, x):
        new_x = []
        for x0 in x:
            # get index of heap element for x0 and data from that element
            index = self.get_index(x0)
            coeff = self.heap[index][1]
            basis = self.heap[index][2]
            order = np.float64(len(coeff) - 1)
            a = self.heap[index][3][0]
            b = self.heap[index][3][1]
            # evaluate the given basis function
            xs = self.basis_function(x0, order, basis, a, b)
            # multiply the calculated basis by their coefficients
            # that are givent for the calculated array
            val = np.dot(np.array(coeff), xs)
            new_x.append(val)
        return np.array(new_x, dtype=np.float64)

    def evaluate(self, x):
        y = []
        for xn in x:
            index = 1
            for i in range(1, self.num_levels-1):
                if self.midpoints[index] > xn:
                    index = 2*index
                else:
                    index = 2*index + 1
            index = self.new_indices[index]
            scale = 2./(self.ranges1[index]-self.ranges0[index])
            xn = scale*(xn - self.ranges0[index]) - 1.0
            xs = self.basis_function(xn, self.max_order, self.basis,
                                     self.ranges0[index], self.ranges1[index])
            y.append(np.dot(self.used_coeff[index], xs))
        return y
