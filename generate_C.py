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
import pyopencl as cl
import pyopencl.array as cl_array


def generate_code(domain_size, ap):
    #c.Statement('for(int n=0; n<' + repr(int(domain_size)) + '; n++)')
    #block = c.Block([c.Statement('return "hello world"'), c.Statement('hi')])
    #declaration = c.FunctionDeclaration(c.Const(c.Value("char", "greet")), [])
    #func = c.FunctionBody(declaration, block)
    the_ifs = []
    for i in range(len(ap.ranges)):
        then = []
        statement = c.Statement('y[n] = ' + repr(ap.coeff[i][ap.orders[i]]))
        then.append(statement)
        for j in range(ap.orders[i])[::-1]:
            # using horner's method, this requires the for loop to be reversed
            statment = c.Statement('y[n] = x[n]*y[n] + ' + repr(ap.coeff[i][j]))
            then.append(statment)
        condition = '(' + repr(ap.ranges[i][0]) + ' <= x[n])'
        condition += ' && (x[n] <= ' + repr(ap.ranges[i][1]) + ')'
        the_if = c.If(condition, c.Block(then))
        the_ifs.append(the_if)
    block = c.Block(the_ifs)
    code = c.For('int n=0', 'n<' + repr(int(domain_size)), 'n++', block)
    return str(code)

"""
def generate_code_legend(domain_size, ap):
    the_ifs = []

    # make legendre array
    # declare base case for recurrence
    legendre = []
    legendre.append(c.Statement('l[0] = 1.0'))
    legendre.append(c.Statement('l[1] = x[n]'))
    for j in range(ap.orders[i]+1):
        one = repr(2.*j-1.) + '*x[n]*l[z-1]' +  ' + '
        two = repr(j-1.) + '*l[z-2]'
        stat = c.Statement('l[z] = ' + one + two)
        legendre.append(stat)
        order = repr(ap.orders[i])
        legendre = c.For('int z=0', 'z<=' + order, 'z++', legendre)

    for i in range(len(ap.ranges)):
        then = []
        # initialize y to 0 
        then.append(c.Statement('y[n] = 0.0'))
        # create the recurrence relation
        for j in range(ap.orders[i]+1):
            rvalue = 'y[n] + l[{0}]*'.format(j) + repr(ap.coeff[i][j])
            line = c.Assign('y[n]', rvalue)
            stat = c.Statement(line )
            then.append(stat)
        condition = '(' + repr(ap.ranges[i][0]) + ' <= x[n])'
        condition += ' && (x[n] <= ' + repr(ap.ranges[i][1]) + ')'
        the_if = c.If(condition, c.Block(then))
        the_ifs.append(the_if)

    # insert legendre array calculation at the begining of the for loop
    legendre.extend(the_ifs)
    block = c.Block(the_ifs)
    a_for = c.For(c.Value('int', 'n=0'), 'n<' + repr(int(domain_size)), 'n++', block)
    l_declaration = "double l[{0}]".format(max(ap.orders)+1)
    code = c.Block([c.Statement(l_declaration), a_for])
    return str(code)
"""

# input is an approximator class
# output is C code.
# simple method for evaluating a monomial interpolant with openCl
# not currently vectorized
def generate_string(domain_size, ap):
    string = "for(int n=0; n<" + repr(int(domain_size)) + "; n++) { "
    # string += "int n = get_global_id(0); "
    for i in range(len(ap.ranges)):
        string += "if ((" + repr(ap.ranges[i][0]) + " <= x[n])"
        string += " && (x[n] <= " + repr(ap.ranges[i][1]) + ")) { "
        string += "y[n] = "
        sub_string = repr(ap.coeff[i][ap.orders[i]])
        for j in range(ap.orders[i])[::-1]:
            # using horner's method, this requires the for loop to be reversed
            sub_string = "x[n]*(" + sub_string + ") + " + repr(ap.coeff[i][j])
        string += sub_string
        string += ";"
        string += "}"
    string += "}"
    return string


# string is executable c code
# x is the co-image of function
def run_c(x, string):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    x_dev = cl_array.to_device(queue, x)
    y_dev = cl_array.empty_like(x_dev)

    # build the code to run from given string
    one = "__kernel void sum(__global double *x,"
    two = " __global double *y) { "
    code = one + two + string + " }"

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
    out = run_c(x, code)
    return out
