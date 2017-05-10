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
import cgen as c
import numpy as np
try:
    with_pyopencl = True
    import pyopencl as cl
    import pyopencl.array as cl_array
except:
    with_pyopencl = False



def gen_mono(ap, domain_size):
    pass


def gen_cheb(ap, domain_size):
    pass


def gen_leg(ap, domain_size):
    pass


# this is approximately 10 times faster than legendre evaluation
# generate string is more than twice as fast as this though.
def gen_mono_b(ap, domain_size):
    the_ifs = []
    order = int(ap.max_order)
    for i in range(len(ap.ranges)):
        then = []
        statement = c.Statement('y[n] = ' + repr(ap.coeff[i][order]))
        then.append(statement)
        scale = repr(np.float64(2)/(ap.upper_bound - ap.lower_bound))
        shift = repr(ap.lower_bound)
        statement = c.Statement('x[n] = {0}*(x[n] - {1})-1.0'.format(scale, shift))
        then.append(statement)
        for j in range(order)[::-1]:
            # using horner's method, this requires the for loop to be reversed
            statement = c.Statement('y[n] = x[n]*y[n] + '+repr(ap.coeff[i][j]))
            then.append(statement)
        condition = '(' + repr(ap.ranges[i][0]) + ' <= x[n])'
        condition += ' && (x[n] <= ' + repr(ap.ranges[i][1]) + ')'
        the_if = c.If(condition, c.Block(then))
        the_ifs.append(the_if)
    block = c.Block(the_ifs)
    code = c.For('int n=0', 'n<' + repr(int(domain_size)), 'n++', block)
    return str(c.Block([code]))


# Branching, non vectorized chebyshev code generation
def gen_cheb_b(ap, domain_size):
    string += "{\ndouble T0, T1, Tn, s;\n"
    string += "for(int n=0; n<{0}; i++)".format(int(domain_size))
    string += "{\n"
    for i in range(len(ap.ranges)):
        string += "\tif ((" + repr(ap.ranges[i][-1][0]) + " <= x[n])"
        string += " && (x[n] <= " + repr(ap.ranges[i][-1][1]) + ")) {\n"
        scale = repr(np.float64(2)/(ap.upper_bound - ap.lower_bound))
        shift = repr(ap.lower_bound)
        statement = c.Statement('x[n] = {0}*(x[n] - {1})-1.0'.format(scale, shift))
        string += "\t\tT0 = 1.0;\n"
        if order > 0:
            string += "\t\tT1 = x[n];\n"
        # initialize the sum
        string += "\t\ts = " + repr(ap.coeff[i][j]) + "*T0;\n"
        if order > 0:
            string += "\t\ts = s + " + repr(ap.coeff[i][j]) + "*T1;\n"
        if order > 1:
            for j in range(2, order+1):
                string += "\t\tTn = 2*x[n]*T1 - T0;\n"
                string += "\t\ts = s + " + repr(ap.coeff[i][j]) + "*Tn;\n"
                string += "\t\tT0 = T1;\n"
                string += "\t\tT1 = Tn;\n"
        string += "\t\ty[n] = s;\n"
        string += "\t}"
    string += "}}"
    return string


# generate C code that evaluates legendre polynomials
# according to the approximator class that is given.
def gen_leg_b(ap, domain_size):
    the_ifs = []
    order = int(ap.max_order)
    for i in range(len(ap.ranges)):
        then = []
        if order > 1:
            then.append(c.Statement('l[1] = x[n]'))
            one = '(2*z-1)*x[n]*l[z-1] + '
            two = '(z-1)*l[z-2]'
            A = repr(1./order)
            stat = c.Statement('l[z] = {0}*('.format(A) + one + two + ')')
            l_for = c.For('int z=2', 'z<=' + repr(order), 'z++', c.Block([stat]))
            then.append(l_for)
        # create the legendre evaluation
        rvalue = ''
        for j in range(order+1):
            if j == 0:
                rvalue += repr(ap.coeff[i][j])
            elif j == 1:
                rvalue += repr(ap.coeff[i][j]) + '*x[n]'
            elif j >= 2:
                rvalue += repr(ap.coeff[i][j]) + '*l[{0}]'.format(j)
            if j != order:
                rvalue += ' + '
        # add legendre polynomial evaluation to code
        scale = repr(np.float64(2)/(ap.upper_bound - ap.lower_bound))
        shift = repr(ap.lower_bound)
        statement = c.Statement('x[n] = {0}*(x[n] - {1})-1.0'.format(scale, shift))
        then.append(statement)
        then.append(c.Statement('y[n] = ' + rvalue))
        condition = '(' + repr(ap.ranges[i][0]) + ' <= x[n])'
        condition += ' && (x[n] <= ' + repr(ap.ranges[i][1]) + ')'
        the_if = c.If(condition, c.Block(then))
        the_ifs.append(the_if)
    block = c.Block(the_ifs)
    a_for = c.For('int n=0', 'n<' + repr(int(domain_size)), 'n++', block)
    # initialize start of legendre array
    l_declaration = 'double l[{0}]'.format(order+1)
    init = c.Statement('l[0] = 1.0')
    code = c.Block([c.Statement(l_declaration), init, a_for])
    return str(code)


# make vectorized monomial code
# all the orders must be the same for this code
# see remez for start of this
def gen_mono(ap, size, vector_width, single=False):
    # maximum possible order of representation
    max_or  = int(ap.max_order)
    if single:
        string  = "for (int n = get_local_id(0); n < " + repr(int(size))
        string += "; i+=" + repr(int(vector_width)) + ") {\n"
    else:
        string  = "int n = get_global_id(0);\n"
    # gives the index of the coefficients to use
    string += "int index = 1;\n"
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
        sub_string = "x[n]*(" + sub_string + \
                     ") + tree[index + {0}]".format(j + 1)
    string += "" + sub_string
    string += ";\n"
    if single:
        string+= "}"
    return string


# generate C code that evaluates chebyshev polynomials
# according to the approximator class that is given.
def gen_cheb(ap, size=0, vector_width=0, single=False):
    # maximum possible order of representation
    order = int(ap.max_order)
     if single:
        string  = "for (int n = get_local_id(0); n < " + repr(int(size))
        string += "; i+=" + repr(int(vector_width)) + ") {\n"
    else:
        string  = "int n = get_global_id(0);\n"
    # gives the index of the coefficients to use
    string += "int index = 0;\n"
    string += "double T0, T1, Tn, a, b, s;\n"
    string += "for(int i=1; i<{0}; i++)".format(int(ap.num_levels))
    string += "{\n\tindex = tree[index] > x[n] ? "
    string += "(int)tree[index+{0}] :".format(4+order)
    string += "(int)tree[index+{0}];\n".format(5+order)
    string += "} \n"
    # rescale variables
    string += "a = tree[index+{0}];\n".format(2+order)
    string += "b = tree[index+{0}];\n".format(3+order)
    string += "x[n] = (2./(b - a))*(x[n] - a) - 1.0;\n"
    string += "T0 = 1.0;\n"
    if order > 0:
        string += "T1 = x[n];\n"
    # initialize the sum
    string += "s = tree[index+1]*T0;\n"
    if order > 0:
        string += "s = s + tree[index+2]*T1;\n"
    if order > 1:
        # calculate order 2 through n
        string += "for (int j = 3; j <=" + repr(order+1) + "; j++) {\n"
        string += "\tTn = 2*x[n]*T1 - T0;\n"
        string += "\ts = s + tree[index + j]*Tn;\n"
        string += "\tT0 = T1;\n"
        string += "\tT1 = Tn;\n"
        string += "}\n"
    string += "y[n] = s;\n"
    if single:
        string += "}"
    return string


# generates vectorized legendre code without branching given an approximator
def gen_leg(ap, size=0, vector_width=0, single=False):
    # maximum possible order of representation
    order = int(ap.max_order)
    if single:
        string  = "for (int n = get_local_id(0); n < " + repr(int(size))
        string += "; i+=" + repr(int(vector_width)) + ") {\n"
    else:
        string  = "int n = get_global_id(0);\n" 
    # gives the index of the coefficients to use
    string += "int index = 0;\n"
    string += "double L0, L1, Ln, a, b, s;\n"
    string += "for(int i=1; i<{0}; i++)".format(int(ap.num_levels))
    string += "{\n\tindex = tree[index] > x[n] ? "
    string += "(int)tree[index+{0}] :".format(4+order)
    string += "(int)tree[index+{0}];\n".format(5+order)
    string += "} \n"
    # rescale variables
    string += "a = tree[index+{0}];\n".format(2+order)
    string += "b = tree[index+{0}];\n".format(3+order)
    string += "x[n] = (2/(b - a))*(x[n] - a) - 1.0;\n"
    string += "L0 = 1.0;\n"
    if order > 0:
        string += "L1 = x[n];\n"
    # initialize the sum
    string += "s = tree[index+1]*L0;\n"
    if order > 0:
        string += "s = s + tree[index+2]*L1;\n"
    if order > 1:
        string += "for (int j = 3; j <=" + repr(order+1) + "; j++) {\n"
        string += "\tLn = ((2.*j-1.)*x[n]*L1 + (j-1.)*L0)/"+repr(np.float64(order))+";\n"
        string += "\ts = s + tree[index + j]*Ln;\n"
        string += "\tL0 = L1;\n"
        string += "\tL1 = Ln;\n"
        string += "}\n"
    string += "y[n] = s;\n"
    if single:
        string += "}"
    return string


# input is an approximator class
# output is C code.
# simple method for evaluating a monomial interpolant with openCl
# not currently vectorized
def gen_mono_vb(ap):
    string = "int n = get_global_id(0);\n"
    for i in range(len(ap.ranges)):
        string += "if ((" + repr(ap.ranges[i][-1][0]) + " <= x[n])"
        string += " && (x[n] <= " + repr(ap.ranges[i][-1][1]) + ")) {\n"
        string += "\ty[n] = "
        sub_string = repr(ap.ranges[i][1][-1])
        for j in range(len(ap.ranges)-1)[::-1]:
            # using horner's method, this requires the for loop to be reversed
            sub_string = "x[n]*(" + sub_string + ") + " \
                          + repr(ap.ranges[i][1][j])
        string += sub_string
        string += ";\n"
        string += "}\n"
    return string


def gen_cheb_vb(ap):
    pass


def gen_leg_vb(ap):
    pass


# runs the code for a normal monomial basis with no transformations.
def run_mono_vec(x, approx):
    if with_pyopencl:
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        queue.finish()

        tree = approx.tree_1d

        x_dev = cl_array.to_device(queue, x)
        table_dev = cl_array.to_device(queue, tree_1d)
        coeff_dev = cl_array.to_device(queue, coeff)
        y_dev = cl_array.empty_like(x_dev)

        # build the code to run from given string
        declaration = "__kernel void sum(__global double *tree, "
        declaration += "__global double *x, __global double *y)"
        code = declaration + '{' + approx.code + '}'

        prg = cl.Program(ctx, code).build()
        # second parameter determines how many 'code instances' to make, (1, ) is 1
        prg.sum(queue, x_dev.shape, None, table_dev.data,
                coeff_dev.data, x_dev.data, y_dev.data)

        queue.finish()


        return y_dev.get()
    else:
        raise ValueError("Function requires pyopencl installation.")


# string is executable c code
# x is the co-image of function
def run_ortho_vec_old(x, approx):
    if with_pyopencl:
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        queue.finish()

        coeff = approx.coeff_1d
        table = approx.midpoints
        ranges0 = approx.ranges0
        ranges1 = approx.ranges1
        new_index = approx.new_indices

        print(coeff)
        print(table)
        print(ranges0)
        print(ranges1)
        print(new_index)


        x_dev = cl_array.to_device(queue, x)
        table_dev = cl_array.to_device(queue, table)
        coeff_dev = cl_array.to_device(queue, coeff)
        ranges0_dev = cl_array.to_device(queue, ranges0)
        ranges1_dev = cl_array.to_device(queue, ranges1)
        new_index_dev = cl_array.to_device(queue, new_index)
        y_dev = cl_array.empty_like(x_dev)

        # build the code to run from given string
        declaration = "__kernel void sum(__global double *mid, "
        declaration += "__global double *coeff, __global double *ranges0, "
        declaration += "__global double *ranges1, __global double *x, "
        declaration += "__global double *y, __global double *new_index) "
        code = declaration + '{' + approx.code + '}'

        prg = cl.Program(ctx, code).build()
        # second parameter determines how many 'code instances' to make, (1, ) is 1
        # should be x_dev.shape
        prg.sum(queue, x_dev.shape, None, table_dev.data,
                coeff_dev.data, ranges0_dev.data, 
                ranges1_dev.data, x_dev.data, y_dev.data, new_index_dev.data)

        queue.finish()

        return y_dev.get()
    else:
        raise ValueError("Function requires pyopencl installation.")


# string is executable c code
# x is the co-image of function
def run_full_code(x, approx, size, vector_width):
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
        # second parameter determines how many 'code instances' to make, (1, ) is 1
        # should be x_dev.shape

        prg.sum(queue,(size,), (vector_width,), tree_dev.data,
                x_dev.data, y_dev.data)

        queue.finish()

        return y_dev.get()
    else:
        raise ValueError("Function requires pyopencl installation.")


# string is executable c code
# x is the co-image of function
def build_code(x, approx, size, vector_width):
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

        return knl, queue, x, y, tree
    else:
        raise ValueError("Function requires pyopencl installation.")


def run_code(knl, queue, x, y, tree, size, vector_width):
    queue.finish()
    start = time.time()
    knl(queue, (int(size),), (int(vector_width),), tree.data, x.data, y.data)
    queue.finish()
    return time.time() - start, y.get()
 


########################################
#                                      #
#    USED FOR TESTING PERFORMANCE      #
#                                      #
########################################

# string is executable c code
# x is the co-image of function
def build_test(x, approx):
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
    # second parameter determines how many 'code instances' to make, (1, ) is 1
    # should be x_dev.shape
    knl = prg.sum
    queue.finish()
    return knl, queue, x_dev, y_dev, tree_dev

def run_test(knl, queue, x, y, tree, vw):
    queue.finish()
    start = time.time()
    knl(queue, (int(vw),), (int(vw),), tree.data, x.data, y.data)
    queue.finish()
    return time.time() - start, y.get()
 

# generate C code that evaluates chebyshev polynomials
# according to the approximator class that is given.
def gen_test(ap, size, vector_width):
    # maximum possible order of representation
    order = int(ap.max_order)
    string = "for (int n=get_global_id(0); n<"+repr(int(size))+";n+="+repr(int(vector_width))+"){"
    #string += "int n = get_global_id(0);\n"
    # gives the index of the coefficients to use
    string += "int index = 0;\n"
    string += "double T0=0;\ndouble T1=0;\ndouble Tn=0;\n"
    string += "double a=0;\ndouble b=0;\ndouble s=0;\ndouble x_scaled;\n"
    string += "for(int i=1; i<{0}; i++)".format(int(ap.num_levels))
    string += "{\n\tindex = tree[index] > x[n] ? "
    string += "(int)tree[index+{0}] : ".format(4+order)
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
    string += "y[n] = s;\n}\n"
    ap.code = string
    return string


