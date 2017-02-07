"""
Create C Code that will evaluate the approximator
class much more quickly.

Code that runs an adaptive interpolation method, then runs it
as C code. I also writes the method
as C code into a file to save if the user wishes to use it later

"""

from __future__ import absolute_import
from __future__ import print_function

import time
import cgen as c
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array


# this is approximately 10 times faster than legendre evaluation
# generate string is more than twice as fast as this though.
def gen_mono_b(ap, domain_size):
    the_ifs = []
    for i in range(len(ap.ranges)):
        then = []
        statement = c.Statement('y[n] = ' + repr(ap.coeff[i][ap.orders[i]]))
        then.append(statement)
        for j in range(ap.orders[i])[::-1]:
            # using horner's method, this requires the for loop to be reversed
            statement = c.Statement('y[n] = x[n]*y[n] + ' + repr(ap.coeff[i][j]))
            then.append(statement)
        condition = '(' + repr(ap.ranges[i][0]) + ' <= x[n])'
        condition += ' && (x[n] <= ' + repr(ap.ranges[i][1]) + ')'
        the_if = c.If(condition, c.Block(then))
        the_ifs.append(the_if)
    block = c.Block(the_ifs)
    code = c.For('int n=0', 'n<' + repr(int(domain_size)), 'n++', block)
    return str(c.Block([code]))


# generate C code that evaluates legendre polynomials 
# according to the approximator class that is given.
def gen_leg_b(ap, domain_size):
    the_ifs = []
    for i in range(len(ap.ranges)):
        then = []
        if ap.orders[i] > 1:
            then.append(c.Statement('l[1] = x[n]'))
            #for j in range(ap.orders[i]+1):
            one = '(2*z-1)*x[n]*l[z-1] + '
            two = '(z-1)*l[z-2]'
            A = repr(1./ap.orders[i])
            stat = c.Statement('l[z] = {0}*('.format(A) + one + two + ')')
            order = repr(ap.orders[i])

            l_for = c.For('int z=2', 'z<=' + order, 'z++', c.Block([stat]))
            then.append(l_for)
        # create the legendre evaluation
        rvalue = ''
        for j in range(ap.orders[i]+1):
            if j == 0:
                rvalue += repr(ap.coeff[i][j])
            elif j == 1:
                rvalue += repr(ap.coeff[i][j]) + '*x[n]' 
            elif j >= 2:
                rvalue += repr(ap.coeff[i][j]) + '*l[{0}]'.format(j)
            if j != ap.orders[i]:
                rvalue += ' + '
        # add legendre polynomial evaluation to code
        then.append(c.Statement('y[n] = ' + rvalue))
        condition = '(' + repr(ap.ranges[i][0]) + ' <= x[n])'
        condition += ' && (x[n] <= ' + repr(ap.ranges[i][1]) + ')'
        the_if = c.If(condition, c.Block(then))
        the_ifs.append(the_if)

    block = c.Block(the_ifs)
    a_for = c.For('int n=0', 'n<' + repr(int(domain_size)), 'n++', block)
    # initialize start of legendre array
    l_declaration = 'double l[{0}]'.format(max(ap.orders)+1)
    init = c.Statement('l[0] = 1.0')
    code = c.Block([c.Statement(l_declaration), init, a_for])
    return str(code)


# generate C code that evaluates chebyshev polynomials 
# according to the approximator class that is given.
def gen_cheb_v(ap):
    # maximum possible order of representation
    order = ap.max_order
    string = "int n = get_global_id(0);"
    # gives the index of the coefficients to use
    string += "int index = 1;"
    string += "double T0, T1, Tn, s;"
    string += "for(int i=1; i<{0}; i++)".format(ap.num_levels)
    string +=     "{ index = mid[index] > x[n] ? 2*index : 2*index+1;"
    string += "} "
    string += "T0 = 1.0;"
    if order > 0:
        string += "T1 = x[n];"
    # initialize the sum
    string += "s = coeff[index*{0}+{1}]*T0 + ".format(order+1, 0)
    if order > 0:
        string += "coeff[index*{0}+{1}]*T1;".format(order+1, 1)
    if order > 1:
        string += "for (int j = 2; j <=" + repr(order) + "; j--) {"
        string +=     "Tn = 2*x[n]*T1 - T0;"
        string +=     "s = s + coeff[index*{0} + j]*Tn;".format(order+1)
        string +=     "T0 = T1;"
        string +=     "T1 = Tn;"
        string += "}"
    string += "y[n] = s;"
    return string



# make vectorized monomial code
# all the orders must be the same for this code
# see remez for start of this
def gen_mono_v(ap):
    # maximum possible order of representation
    max_or = ap.max_order
    string = "int n = get_global_id(0);"
    # gives the index of the coefficients to use
    string += "int index = 1;"
    string += "for(int i=1; i<{0}; i++)".format(ap.num_levels)
    string += "{ index = mid[index] > x[n] ? 2*index : 2*index+1;"
    string += "} "
    string += "y[n] = "
    # the coefficients are transformed from a matrix to a vector.
    # the formula to call the correct entry is given as the indices
    sub_string = "coeff[index*{0}+{1}]".format(max_or+1, max_or)
    for j in range(max_or)[::-1]:
        # using horner's method, this requires the for loop to be reversed
        sub_string = "x[n]*(" + sub_string + ") + coeff[index*{0}+{1}]".format(max_or+1, j)
    string += sub_string
    string += ";"
    return string


# input is an approximator class
# output is C code.
# simple method for evaluating a monomial interpolant with openCl
# not currently vectorized
def gen_mono_vb(domain_size, ap):
    #string = "{ for(int n=0; n<" + repr(int(domain_size)) + "; n++) { "
    string = "{ int n = get_global_id(0);"
    for i in range(len(ap.ranges)):
        string += "if ((" + repr(ap.ranges[i][-1][0]) + " <= x[n])"
        string += " && (x[n] <= " + repr(ap.ranges[i][-1][1]) + ")) { "
        string += "y[n] = "
        sub_string = repr(ap.ranges[i][1][-1])
        for j in range(len(ap.ranges)-1)[::-1]:
            # using horner's method, this requires the for loop to be reversed
            sub_string = "x[n]*(" + sub_string + ") + " + repr(ap.ranges[i][1][j])
        string += sub_string
        string += ";"
        string += "}"
    string += "}"# }"
    return string


# string is executable c code
# x is the co-image of function
def run_c(x, string):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    x_dev = cl_array.to_device(queue, x)
    y_dev = cl_array.empty_like(x_dev)

    # build the code to run from given string
    declaration = "__kernel void sum(__global double *x, "
    declaration += "__global double *y) "
    code = declaration + string

    start = time.time()
    prg = cl.Program(ctx, code).build()
    print('Time to run C Code, ', time.time() - start)
    #second parameter determines how many 'code instances' to make, (1, ) is 1
    prg.sum(queue, x_dev.shape, None, x_dev.data, y_dev.data)
    return y_dev.get()


# string is executable c code
# x is the co-image of function
def run_c_v(x, table, coeff, string):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # turn coeff into 1 dimensional numpy array    
    max_order = len(coeff[1])
    coeff_n = np.ones(len(coeff)*max_order).astype(np.float64)
    for i in range(len(coeff)*max_order):
        coeff_n[i] = coeff[i/max_order][i%max_order]


    x_dev = cl_array.to_device(queue, x)
    table_dev = cl_array.to_device(queue, np.array(table).astype(np.float64))
    coeff_dev = cl_array.to_device(queue, coeff_n)
    y_dev = cl_array.empty_like(x_dev)

    # build the code to run from given string
    declaration = "__kernel void sum(__global double *mid, "
    declaration += "__global double *coeff, __global double *x, "
    declaration += "__global double *y) "
    code = declaration + '{' + string + '}'

    start = time.time()
    prg = cl.Program(ctx, code).build()
    print('Time to run C Code, ', time.time() - start)
    #second parameter determines how many 'code instances' to make, (1, ) is 1
    prg.sum(queue, x_dev.shape, None, table_dev.data, coeff_dev.data, x_dev.data, y_dev.data)
    return y_dev.get()
    
    
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
