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

import pickle
import numpy as np


# class to evaluate an adaptive interpolant that has different
# coefficients on different ranges
class Approximator(object):

    def __init__(self, my_adapt=None, optimizations=[]):
        #print("Building Approximator")
        if my_adapt != None:
            # raw array data from adaptive interpolation method
            self.basis_function = my_adapt.basis_function
            self.tree = my_adapt.tree
            self.error = my_adapt.allowed_error
            self.upper_bound = my_adapt.upper_bound
            self.lower_bound = my_adapt.lower_bound
            self.basis = my_adapt.basis
            self.midpoints = []

            # arrays used for code generation
            self.mid = []
            self.coeff = []
            self.left = [0] * self.tree.size
            self.right = [0] * self.tree.size
            self.interval_a = []
            self.interval_b = []
            self.map = []
            self.cmap = []

            # used for animation of algorithm
            self.used_midpoints = []
            self.rel_errors = []
            self.used_errors = []
            self.ranges = []
            self.used_ranges = []

            self.curr_index = -1
            self.leaf_index = -1
            self.max_order = int(my_adapt.max_order)
            self.num_levels = self.tree.max_level+1
            self.tree_vector = [None] * (self.tree.size)

            self.optimizations = optimizations
            self.D = 0
            self.lgD = 0
            self.D_ones = 0
            self.lgD_ones = 0

            if "map" in optimizations:
                self.make_tree_array(self.tree.root)
            else:
                self.make_tree_array(self.tree.root)

            # these values are used for code generation and evaluation
            self.tree_1d = np.array(self.tree_1d(), dtype=my_adapt.dtype)
            self.code = 0
            self.size = None
            self.vector_width = None
            self.dtype = my_adapt.dtype
            self.dtype_name = ""
            # num function outputs is equal to number of
            # leaves if the tree we complete

            self.kernal = None
            self.queue = None
            self.x_dev = None
            self.y_dev = None
            self.tree_dev = None

            """
            print(self.mid)
            print(self.left)
            print(self.right)
            print(self.interval_a)
            print(self.interval_b)
            print(self.coeff)
            print(self.map)

            print(len(self.map), 2**self.tree.max_level)
            """
            #print(len(self.coeff), len(self.map))
            #print(self.map[int(3*1.6)], self.map[int(20*1.6)])
            #print(self.map[int(3*1.6)] & 2**20 -1, int(19*1.6))
            self.intervals = self.interval_a + [self.interval_b[-1]]
            assert len(self.intervals) == len(self.interval_a) + 1
            if "verbose" in self.optimizations:
                print("Map: ", self.leaf_index+1, self.map)
                print("A: ", self.interval_a)
                print("B: ", self.interval_b)
                print("coeffs: ", self.coeff)

            if "map" in self.optimizations:
                assert len(self.map) == 2**self.tree.max_level
                assert len(self.cmap) == 2**self.tree.max_level

            print("Index Storage Cost: ", 2*self.D + self.lgD)
            dt = np.int32 if (2*self.D + self.lgD) < 32 else np.int64
            self.cmap = np.array(self.cmap, dtype=dt)
            
            self.map = np.array(self.map, dtype=np.int32)



    def tree_1d(self):
        tree_1d = []
        # tree = [mid, coeff0, coeff1, .., coeffn,
        #         a, b, left_index, right_index, ...]
        # with optimizations
        # tree = [mid, left_index, right_index, coeff0, coeff1, .., coeffn,
        #         a, b, ...]
        #print(self.tree_vector)
        for node in self.tree_vector:
            # if not leaf, only three objects and each will be added
            # by the elif statement
            for i in range(len(node)):
                if type(node[i]) == list or type(node[i]) == np.ndarray:
                    for j in range(len(node[i])):
                        tree_1d.append(node[i][j])
                else:
                    tree_1d.append(node[i])
        #print(tree_1d)
        return tree_1d



    # data = [mid, coeff, range, error]
    def convert_to_arrays(self, node):
        self.midpoints.append(node.data[0])
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


    def store_interval_info_in_map(self, node):
        # root is D=0, this should not have +1 but without not high enough. something wrong somewhere else
        D = self.tree.max_level+1
        lgD = int(np.log2(D))+1
        dt = np.int32 if (2*D + lgD) < 32 else np.int64
        self.D = D
        self.lgD = lgD

        D_ones = dt(2**D - 1)
        lgD_ones = dt(2**lgD - 1)
        self.D_ones = D_ones
        self.lgD_ones = lgD_ones

        num_leaves_for_node = 2**(self.tree.max_level - node.get_level())
        length = len(self.cmap)
        l = dt(length)
        r = dt(length + num_leaves_for_node)
        L = r - l
        leafuint = dt(self.leaf_index)

        # each of these numbers should only be
        assert D_ones & l == l
        assert D_ones & r == r
        assert lgD_ones & L == L
        assert D_ones & leafuint == leafuint
        num = dt(0)
        num ^= leafuint
        num ^= (l <<   D)
        num ^= (L << 2*D)
        #print( bin(num & D_ones), bin(leafuint))
        #print( bin((num >> D) & D_ones), bin(l))
        #print( bin((num >> 2*D) & D_ones), bin(r))
        #print(leafuint,l, r, bin(num))
        return num


    def make_tree_array(self, node):
        data_size = 3 + len(node.data[1]) + len(node.data[2])
        self.curr_index += 1
        my_index = self.curr_index
        # if leaf node then make it it's own child
        if node.left == 0 or node.right == 0 or node.left == node or node.right == node:
            self.leaf_index += 1
            data = [my_index*data_size, my_index*data_size]
            for i in range(len(node.data)):
                # don't add error term from node into data
                # if using td opt then order data as [mid, right, left, coeff, ...]
                if "trim_data" in self.optimizations:
                    if i == 0:
                        data.insert(0, node.data[i])
                    elif i != 3:
                        data.append(node.data[i])
                # otherwise [mid coeff, interval, right, left, ...]
                else:
                    if i != 3:
                        data.insert(i, node.data[i])
            if 'arrays' in self.optimizations:
                self.mid.append(node.data[0])
                for i in range(len(node.data[1])):
                    self.coeff.append(node.data[1][i])
                self.interval_a.append(node.data[2][0])
                self.interval_b.append(node.data[2][1])
                # needs to be a function to map this to leaf index if not using map opt
                self.left[my_index] = my_index
                self.right[my_index] = my_index

                # used for calculating intervals
                num = self.store_interval_info_in_map(node)
                num_leaves_for_node = 2**(self.tree.max_level - node.get_level())
                for i in range(num_leaves_for_node):
                    self.cmap.append(num)
                    self.map.append(self.leaf_index)
            self.tree_vector[my_index] = data
            return my_index
        left_index = self.make_tree_array(node.left)
        right_index = self.make_tree_array(node.right)
        # if using td opt also only leaves will contain their coeff data
        if "trim_data" in self.optimizations:
            self.tree_vector[my_index] = [node.data[0], left_index, right_index]
        else:
            data = [left_index*data_size, right_index*data_size]
            for i in range(len(node.data)):
                if i != 3:
                    data.insert(i, node.data[i]) 
            self.tree_vector[my_index] = data
        if "arrays" in self.optimizations:
            self.left[my_index] = left_index
            self.right[my_index] = right_index
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
        if "trim_data" in self.optimizations:
            coeff_index = 3
            interval_index = self.max_order + 4
            child_index = 1
        else:
            coeff_index = 1
            interval_index = self.max_order + 2
            child_index = self.max_order + 4
        #print(self.tree_1d)
        for xn in x:
            index = 0
            for i in range(1, self.num_levels):
                if self.tree_1d[index] > xn:#midpoints
                    index = int(self.tree_1d[index+child_index])
                else:
                    index = int(self.tree_1d[index+child_index+1])
            a = self.tree_1d[index+interval_index]
            b = self.tree_1d[index+interval_index+1]
            xs = self.basis_function(xn, self.max_order, self.basis, a, b)
            coeff = self.tree_1d[index+coeff_index:index+coeff_index+self.max_order+1]
            y.append(np.dot(coeff, xs))
        return y

