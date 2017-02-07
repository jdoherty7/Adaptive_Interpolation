"""
class that takes calculated approximation and
allows adaptive interpolant to be evaluated

A cleaner and more efficient rewrite
of simpleAdapt.py

Will also easily implement different node choices,
interpolant choices, etc.
"""
import numpy as np

# class to evaluate an adaptive interpolant that has different
# coefficients on different ranges
class Approximator(object):

    def __init__(self, adapt):
        # raw array data from adaptive interpolation method
        self.adapt = adapt
        self.heap = adapt.heap
        self.basis = adapt.basis
        self.code
        self.max_order = adapt.max_order
        self.num_levels = int(np.log(len(self.heap))/np.log(2))
        self.midpoints, self.coeff = self.make_trees()
        self.ranges = self.heap[-2**(self.num_levels-1):]

    def make_trees(self):
        midpoints = [0]
        # initialize first element as coeff vector of 0 values
        # assume that all coeff vectors are numpy arrays in heap
        coeff = [0*self.heap[1][1]]
        # assume that the heap is properly allocated
        for i in range(1, len(self.heap)):
            midpoints.append(self.heap[i][0])
            coeff.append(self.heap[i][1])
        return midpoints, coeff

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
            order = len(coeff) - 1
            # evaluate the given basis function
            xs = self.adapt.basis_function(x0, order, basis)
            # multiply the calculated basis by their coefficients
            # that are givent for the calculated array
            val = np.dot(np.array(coeff), xs)
            new_x.append(val)
        return new_x

"""
x = np.outer(np.arange(1,6), np.arange(1,4))
y = 0*x
x_new = np.ones(len(x)*len(x[1])).astype(np.float64)
for i in range(len(x)*len(x[1])):
    x_new[i] = x[i/len(x[1])][i%len(x[1])]

print x
print x_new

max_order = x.shape[1]
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        y[i, j] = x_new[i*(max_order) + j]
print y
"""
