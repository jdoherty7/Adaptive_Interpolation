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
        shift = repr(ap.lower_bound + 1)
        statement = c.Statement('{0}*x[n] - {1}'.format(scale, shift))
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
        shift = repr(ap.lower_bound + 1)
        string += "\t\tx[n] = {0}*x[n] - {1};\n".format(scale, shift)
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
        shift = repr(ap.lower_bound + 1)
        statement = c.Statement('{0}*x[n] - {1}'.format(scale, shift))
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
def gen_mono_v(ap):
    # maximum possible order of representation
    max_or = int(ap.max_order)
    string = "int n = get_global_id(0);\n"
    # gives the index of the coefficients to use
    string += "int index = 1;\n"
    string += "for(int i=1; i<{0}; i++)".format(int(ap.num_levels))
    string += "{ \n\tindex = mid[index] > x[n] ? 2*index : 2*index+1;\n"
    string += "} \n"
    scale = repr(np.float64(2)/(ap.upper_bound - ap.lower_bound))
    shift = repr(ap.lower_bound + 1)
    string += "x[n] = {0}*x[n] - {1};\n".format(scale, shift)
    string += "y[n] = "
    # the coefficients are transformed from a matrix to a vector.
    # the formula to call the correct entry is given as the indices
    sub_string = "coeff[index*{0}+{1}]".format(max_or+1, max_or)
    for j in range(max_or)[::-1]:
        # using horner's method, this requires the for loop to be reversed
        sub_string = "x[n]*(" + sub_string + \
                     ") + coeff[index*{0}+{1}]".format(max_or+1, j)
    string += sub_string
    string += ";\n"
    return string


# generate C code that evaluates chebyshev polynomials
# according to the approximator class that is given.
def gen_cheb_v(ap):
    # maximum possible order of representation
    order = int(ap.max_order)
    string = "int n = get_global_id(0);\n"
    # gives the index of the coefficients to use
    string += "int index = 1;\n"
    string += "double T0, T1, Tn, s;\n"
    string += "for(int i=1; i<{0}; i++)".format(int(ap.num_levels))
    string += "{\n\tindex = mid[index] > x[n] ? 2*index : 2*index+1;\n"
    string += "} \n"
    scale = repr(np.float64(2)/(ap.upper_bound - ap.lower_bound))
    shift = repr(ap.lower_bound + 1)
    string += "x[n] = {0}*x[n] - {1};\n".format(scale, shift)
    string += "T0 = 1.0;\n"
    if order > 0:
        string += "T1 = x[n];\n"
    # initialize the sum
    string += "s = coeff[index*{0}+{1}]*T0;\n".format(order+1, 0)
    if order > 0:
        string += "s = s + coeff[index*{0}+{1}]*T1;\n".format(order+1, 1)
    if order > 1:
        string += "for (int j = 2; j <=" + repr(order) + "; j++) {\n"
        string += "\tTn = 2*x[n]*T1 - T0;\n"
        string += "\ts = s + coeff[index*{0} + j]*Tn;\n".format(order+1)
        string += "\tT0 = T1;\n"
        string += "\tT1 = Tn;\n"
        string += "}\n"
    string += "y[n] = s;\n"
    return string


# generates vectorized legendre code without branching given an approximator
def gen_leg_v(ap):
    # maximum possible order of representation
    order = int(ap.max_order)
    string = "int n = get_global_id(0);\n"
    # gives the index of the coefficients to use
    string += "int index = 1;\n"
    string += "double L0, L1, Ln, s;\n"
    string += "for(int i=1; i<{0}; i++)".format(ap.num_levels)
    string += "{\n\tindex = mid[index] > x[n] ? 2*index : 2*index+1;\n"
    string += "}\n"
    scale = repr(np.float64(2)/(ap.upper_bound - ap.lower_bound))
    shift = repr(ap.lower_bound + 1)
    string += "x[n] = {0}*x[n] - {1};\n".format(scale, shift)
    string += "L0 = 1.0;\n"
    if order > 0:
        string += "L1 = x[n];\n"
    # initialize the sum
    string += "s = coeff[index*{0}+{1}]*L0;\n".format(order+1, 0)
    if order > 0:
        string += "s = s + coeff[index*{0}+{1}]*L1;\n".format(order+1, 1)
    if order > 1:
        string += "for (int j = 2; j <=" + repr(order) + "; j++) {\n"
        string += "\tLn = ((2.*j-1.)*x[n]*L1 - (j-1.)*L0)/"+repr(order)+";\n"
        string += "\ts = s + coeff[index*{0} + j]*Ln;\n".format(order+1)
        string += "\tL0 = L1;\n"
        string += "\tL1 = Ln;\n"
        string += "}\n"
    string += "y[n] = s;\n"
    return string


# input is an approximator class
# output is C code.
# simple method for evaluating a monomial interpolant with openCl
# not currently vectorized
def gen_mono_vb(ap):
    string = "{ int n = get_global_id(0);\n"
    for i in range(len(ap.ranges)):
        string += "if ((" + repr(ap.ranges[i][-1][0]) + " <= x[n])"
        string += " && (x[n] <= " + repr(ap.ranges[i][-1][1]) + ")) {\n"
        scale = repr(np.float64(2)/(ap.upper_bound - ap.lower_bound))
        shift = repr(ap.lower_bound + 1)
        string += "x[n] = {0}*x[n] - {1};\n".format(scale, shift)
        string += "\ty[n] = "
        sub_string = repr(ap.ranges[i][1][-1])
        for j in range(len(ap.ranges)-1)[::-1]:
            # using horner's method, this requires the for loop to be reversed
            sub_string = "x[n]*(" + sub_string + ") + " \
                          + repr(ap.ranges[i][1][j])
        string += sub_string
        string += ";\n"
        string += "}\n"
    string += "}\n"
    return string


def gen_cheb_vb(ap):
    pass


def gen_leg_vb(ap):
    pass


# string is executable c code
# x is the co-image of function
def run_c(x, approx):
    if with_pyopencl:
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        x_dev = cl_array.to_device(queue, x)
        y_dev = cl_array.empty_like(x_dev)

        # build the code to run from given string
        declaration = "__kernel void sum(__global double *x, "
        declaration += "__global double *y) "
        code = declaration + approx.code

        prg = cl.Program(ctx, code).build()

        # second parameter determines how many 'code instances' to make, (1, ) is 1
        prg.sum(queue, x_dev.shape, None, x_dev.data, y_dev.data)
        return y_dev.get()
    else:
        raise ValueError("Function requires pyopencl installation.")


# string is executable c code
# x is the co-image of function
def run_c_v(x, approx):
    if with_pyopencl:
        coeff = approx.coeff_1d
        table = approx.midpoints
        max_order = approx.max_order


        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        x_dev = cl_array.to_device(queue, x)
        table_dev = cl_array.to_device(queue, table)
        coeff_dev = cl_array.to_device(queue, coeff)
        y_dev = cl_array.empty_like(x_dev)

        # build the code to run from given string
        declaration = "__kernel void sum(__global double *mid, "
        declaration += "__global double *coeff, __global double *x, "
        declaration += "__global double *y) "
        code = declaration + '{' + approx.code + '}'

        prg = cl.Program(ctx, code).build()
        # second parameter determines how many 'code instances' to make, (1, ) is 1
        prg.sum(queue, x_dev.shape, None, table_dev.data,
                coeff_dev.data, x_dev.data, y_dev.data)

        return y_dev.get()
    else:
        raise ValueError("Function requires pyopencl installation.")


# use to save the generated code for later use
def write_to_file(file_name, string):
    my_file = open("generated_code/"+file_name+".txt", "w")
    my_file.write(string)
    my_file.close()


# run the C method from the given file using the given domain
def Run_from_file(x, file_name):
    my_file = open("generated_code/"+file_name+".txt", "r")
    # code is just on first line of file, so get this then run it
    code = my_file.readline()
    out = run_c(x, code)
    return out

