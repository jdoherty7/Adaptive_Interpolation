"""
Create C Code that will evaluate the approximator
class much more quickly.

Code that runs an adaptive interpolation method, then runs it
as C code. I also writes the method
as C code into a file to save if the user wishes to use it later

"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import numpy as np
try:
    with_pyopencl = True
    import pyopencl as cl
    import pyopencl.array as cl_array
except:
    with_pyopencl = False


# make vectorized monomial code
# all the orders must be the same for this code
# see remez for start of this
def gen_mono(ap):
    single = False if (ap.size is None and ap.vector_width is None) else True
    print("MONO GENERATE: ", single)
    # maximum possible order of representation
    max_or  = int(ap.max_order)
    if single:
        string  = "for (int n = get_global_id(0); n < " + repr(int(ap.size))
        string += "; n+=" + repr(int(ap.vector_width)) + ") {\n"
    else:
        string  = "int n = get_global_id(0);\n"
    # gives the index of the coefficients to use
    string += "int index = 1;\n"
    string += "double x_const = x[n];\n"
    string += "for(int i=1; i<{0}; i++)".format(int(ap.num_levels))
    string += "{\n\tindex = tree[index] > x[n] ? "
    string += "(int)tree[index+{0}] :".format(4+order)
    string += "(int)tree[index+{0}];\n".format(5+order)
    string += "}\n"
    string += "y[n] = "
    # the coefficients are transformed from a matrix to a vector.
    # the formula to call the correct entry is given as the indices
    sub_string = "tree[index + {0}]".format(max_or + 2)
    for j in range(max_or)[::-1]:
        # using horner's method, this requires the for loop to be reversed
        sub_string = "x_const*(" + sub_string + \
                     ") + tree[index + {0}]".format(j + 1)
    string += "" + sub_string
    string += ";\n"
    if single:
        string+= "}"
    return string


# generate C code that evaluates chebyshev polynomials
# according to the approximator class that is given.
def gen_cheb(ap):
    single = False if (ap.size is None and ap.vector_width is None) else True
    # maximum possible order of representation
    order = int(ap.max_order)
    if single:
        string  = "for (int n = get_global_id(0); n < " + repr(int(ap.size))
        string += "; n+=" + repr(int(ap.vector_width)) + ") {\n"
    else:
        string  = "int n = get_global_id(0);\n"
    # gives the index of the coefficients to use
    string += "int index = 0;\n"
    string += "double T0, T1, Tn, a, b, s, x_scaled;\n"
    string += "for(int i=1; i<{0}; i++)".format(int(ap.num_levels))
    string += "{\n\tindex = tree[index] > x[n] ? "
    string += "(int)tree[index+{0}] :".format(4+order)
    string += "(int)tree[index+{0}];\n".format(5+order)
    string += "} \n"
    # rescale variables
    string += "a = tree[index+{0}];\n".format(2+order)
    string += "b = tree[index+{0}];\n".format(3+order)
    string += "x_scaled = (2./(b - a))*(x[n] - a) - 1.0;\n"
    string += "T0 = 1.0;\n"
    if order > 0:
        string += "T1 = x_scaled;\n"
    # initialize the sum
    string += "s = tree[index+1]*T0;\n"
    if order > 0:
        string += "s = s + tree[index+2]*T1;\n"
    if order > 1:
        # calculate order 2 through n
        string += "for (int j = 3; j <=" + repr(order+1) + "; j++) {\n"
        string += "\tTn = 2*x_scaled*T1 - T0;\n"
        string += "\ts = s + tree[index + j]*Tn;\n"
        string += "\tT0 = T1;\n"
        string += "\tT1 = Tn;\n"
        string += "}\n"
    string += "y[n] = s;\n"
    if single:
        string += "}"
    return string


# generates vectorized legendre code without branching given an approximator
def gen_leg(ap):
    single = False if (ap.size is None and ap.vector_width is None) else True
    # maximum possible order of representation
    order = int(ap.max_order)
    if single:
        string  = "for (int n = get_global_id(0); n < " + repr(int(ap.size))
        string += "; n+=" + repr(int(ap.vector_width)) + ") {\n"
    else:
        string  = "int n = get_global_id(0);\n" 
    # gives the index of the coefficients to use
    string += "int index = 0;\n"
    string += "double L0, L1, Ln, a, b, s, x_scaled;\n"
    string += "for(int i=1; i<{0}; i++)".format(int(ap.num_levels))
    string += "{\n\tindex = tree[index] > x[n] ? "
    string += "(int)tree[index+{0}] :".format(4+order)
    string += "(int)tree[index+{0}];\n".format(5+order)
    string += "} \n"
    # rescale variables
    string += "a = tree[index+{0}];\n".format(2+order)
    string += "b = tree[index+{0}];\n".format(3+order)
    string += "x_scaled = (2/(b - a))*(x[n] - a) - 1.0;\n"
    string += "L0 = 1.0;\n"
    if order > 0:
        string += "L1 = x_scaled;\n"
    # initialize the sum
    string += "s = tree[index+1]*L0;\n"
    if order > 0:
        string += "s = s + tree[index+2]*L1;\n"
    if order > 1:
        string += "for (int j = 3; j <=" + repr(order+1) + "; j++) {\n"
        string += "\tLn = ((2.*j-1.)*x_scaled*L1 + (j-1.)*L0)/"+repr(np.float64(order))+";\n"
        string += "\ts = s + tree[index + j]*Ln;\n"
        string += "\tL0 = L1;\n"
        string += "\tL1 = Ln;\n"
        string += "}\n"
    string += "y[n] = s;\n"
    if single:
        string += "}"
    return string



###############################################
#                                             #
#      METHOD FOR RUNNING GENERATED CODE      #
#                                             #
###############################################


# build and run generated code all in one time
# returns calculated values of resulting functions
# only used to run multithreaded codes
def run(x, approx):
    if with_pyopencl:
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        queue.finish()
        tree = approx.tree_1d

        # initialize variables
        x_dev = cl_array.to_device(queue, x)
        tree_dev = cl_array.to_device(queue, tree)
        y_dev = cl_array.empty_like(x_dev)

        # build the code to run from given string
        declaration = "__kernel void sum(__global double *tree, "
        declaration += "__global double *x, __global double *y) "
        code = declaration + '{' + approx.code + '}'

        # compile code and then execute it
        prg = cl.Program(ctx, code).build()
        prg.sum(queue, x_dev.shape, None, tree_dev.data, 
                x_dev.data, y_dev.data)

        queue.finish()

        return y_dev.get()
    else:
        raise ValueError("Function requires pyopencl installation.")


# builds a pyopencl kernal object that can be used to 
# run the generated and built code multiple times with different inputs
def build_code(x, approx):
    if with_pyopencl:
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        queue.finish()
        tree = approx.tree_1d

        x_dev = cl_array.to_device(queue, x)
        tree_dev = cl_array.to_device(queue, tree)
        y_dev = cl_array.empty_like(x_dev)

        # build the code to run from given string
        declaration = "__kernel void sum(__global double *tree, "
        declaration += "__global double *x, __global double *y) "
        code = declaration + '{' + approx.code + '}'

        prg = cl.Program(ctx, code).build()
        knl = prg.sum
        queue.finish()

        approx.kernal   = knl
        approx.queue    = queue
        approx.x_dev    = x_dev
        approx.y_dev    = y_dev
        approx.tree_dev = tree_dev

        return knl, queue, x_dev, y_dev, tree_dev
    else:
        raise ValueError("Function requires pyopencl installation.")


def run_single(ap):
    ap.queue.finish()
    start = time.time()
    ap.kernal(ap.queue, (int(ap.vector_width),), (int(ap.vector_width),),
              ap.tree_dev.data, ap.x_dev.data, ap.y_dev.data)
    ap.queue.finish()
    end = time.time() - start
    return end, ap.y_dev.get()


def run_multi(ap):
    ap.queue.finish()
    start = time.time()
    ap.knl(ap.queue, ap.x_dev.shape, None, ap.tree_dev.data, ap.x_dev.data, ap.y_dev.data)
    ap.queue.finish()
    end = time.time() - start
    return end, ap.y_dev.get()


