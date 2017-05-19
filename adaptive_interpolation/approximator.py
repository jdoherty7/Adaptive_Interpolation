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
            self.tree = my_adapt.tree
            self.upper_bound = my_adapt.upper_bound
            self.lower_bound = my_adapt.lower_bound
            self.basis = my_adapt.basis
            self.midpoints = []
            self.used_midpoints = []
            self.rel_errors = []
            self.used_errors = []
            self.ranges = []
            self.used_ranges = []
            self.curr_index = -1
            self.max_order = int(my_adapt.max_order)
            self.num_levels = self.tree.max_level+1
            self.tree_vector = [0] * (self.tree.size)
            self.make_tree_array(self.tree.root)

            # these values are used for code generation and evaluation
            self.tree_1d = np.array(self.tree_1d(), dtype=my_adapt.dtype)
            self.code = 0
            self.size = None
            self.vector_width = None
            self.dtype = my_adapt.dtype

            # for evaluation
            self.kernal = None
            self.queue = None
            self.x_dev = None
            self.y_dev = None
            self.tree_dev = None


    def tree_1d(self):
        tree_1d = []
        # tree = [mid, coeff0, coeff1, .., coeffn,
        #         a, b, left_index, right_index, ...]
        for node in self.tree_vector:
            if not isinstance(node, (float, int, np.float32, np.float64, np.float16)):
                for i in range(len(node)):
                    if not isinstance(node[i], (float, int, np.float32, np.float64, np.float16)):
                        for j in range(len(node[i])):
                            if not isinstance(node[i][j], (float, int, np.float32, np.float64, np.float16)):
                                for k in range(len(node[i][j])):
                                    tree_1d.append(node[i][j][k])
                            elif j != 3: #dont add error into struct
                                tree_1d.append(node[i][j])
                else:
                    tree_1d.append(node[i]*(self.max_order+6))
        return tree_1d

    # data = [mid, coeff, range, error]
    def convert_to_arrays(self, node):
        self.midpoints.append(node.data[0])
        self.ranges.append(node.data[2])
        self.rel_errors.append(node.data[3])
        if node.left == 0 and node.right == 0:
            self.used_midpoints.append(node.data[0])
            self.used_ranges.append(node.data[2])
            #self.ranges0.append(node.data[2][0])
            #self.ranges1.append(node.data[2][1])
            self.used_errors.append(node.data[3])
        else:
            self.convert_to_arrays(node.left)
            self.convert_to_arrays(node.right)

    def make_tree_array(self, node):
        self.curr_index += 1
        my_index = self.curr_index
        if node.left == 0 or node.right == 0 or node.left == node or node.right == node:
            self.tree_vector[my_index] = [node.data, my_index, my_index]
            return my_index
        left_index = self.make_tree_array(node.left)
        right_index = self.make_tree_array(node.right)
        self.tree_vector[my_index] = [node.data, left_index, right_index]
        return my_index


    def evaluate_tree(self, x, levels=-1, extra=-1):
        if levels==-1:
            levels = self.num_levels
        y = []
        dat = []
        for xn in x:
            node = self.tree.root
            for _ in range(1, int(levels)):
                if node.data[0] > xn:
                    node = node.left if (node.left != 0) else node
                else:
                    node = node.right if (node.right != 0) else node
            xs = self.basis_function(xn, self.max_order, self.basis,
                                     node.data[2][0], node.data[2][1])
            y.append(np.dot(node.data[1], xs))
            if extra != -1:
                if len(dat) > 0:
                    if dat[-1] != node.data:
                        dat.append(node.data)
                else:
                    dat.append(node.data)
        if extra != -1:
            return y, dat
        else:  
            return y

    def evaluate(self, x):
        y = []
        for xn in x:
            index = 0
            for i in range(1, self.num_levels):
                if self.tree_1d[index] > xn:#midpoints
                    index = int(self.tree_1d[index+4+self.max_order])
                else:
                    index = int(self.tree_1d[index+5+self.max_order])
            a = self.tree_1d[index+2+self.max_order]
            b = self.tree_1d[index+3+self.max_order]
            xs = self.basis_function(xn, self.max_order, self.basis, a, b)
            coeff = self.tree_1d[index+1:index+1+self.max_order+1]
            y.append(np.dot(coeff, xs))
        return y

