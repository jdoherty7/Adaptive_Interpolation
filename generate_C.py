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
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array


# input is an approximator class
# output is C code.
# simple method for evaluating a monomial interpolant with openCl
# not currently vectorized
def generate_string(domain_size, ap):
    string = ""
    string += "for(int n=0; n<{0}; n++)".format(int(domain_size))
    string += " { "
    # string += "int n = get_global_id(0); "
    for i in range(len(ap.ranges)):
        string += "if (({0} <= x[n]) && (x[n] <= {1})) { ".format(ap.ranges[i][0], ap.ranges[i][1])
        string += "y[n] = "
        sub_string = "{0:.200f}".format(ap.coeff[i][ap.orders[-1]])
        for j in range(ap.orders[i])[::-1]:
            # using horner's method, this requires the for loop to be reversed
            sub_string = "x[n]*(" + sub_string + ") + {0:.200f}".format(ap.coeff[i][j])
        string += sub_string
        string += ";"
        string += "}"
    string += "}"
    return string


# string is executable c code
# x is the co-image of function
def Run_C(x, string):
    # if 64 this will not work
    x = x.astype(np.float32)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    x_dev = cl_array.to_device(queue, x)
    y_dev = cl_array.empty_like(x_dev)

    # build the code to run from given string
    one = "__kernel void sum(__global float *x,"
    two = " __global float *y) {"
    code = one + two + string + "}"

    start = time.time()
    prg = cl.Program(ctx, code).build()
    print('Time to run C Code, ', time.time() - start)

    prg.sum(queue, (1,), None, x_dev.data, y_dev.data)
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
    out = Run_C(x, code)
    return out
