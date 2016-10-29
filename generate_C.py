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
import pyopencl as cl
import pyopencl.array as cl_array

#input is an approximator class
#output is C code.


#first do simple thing
#make if statements for all ranges and evaluate them for just monomials
#doing this with a string.
def generate_string(domain_size, ap):
    string  = ""
    string += "float curr; "
    string += "for(int n=0; n<{0}; ++n)".format(int(domain_size))
    string += " { "
    string += "curr = x[n];"
    for i in range(len(ap.ranges)):
        string += "if (({0} <= curr) && (curr <= {1}))".format(ap.ranges[i][0], ap.ranges[i][1])
        string += "{ "
        string += "y[n] = "
        for j in range(ap.order):
            #power function is defined outside of __kernal in Run_C
            string += "{0}*power(curr, {1})".format(ap.coeff[i][j], j)
            if j != ap.order-1:
                string += " + "
        string += ";"
        string += "}"
    string += "}"        
    return string

#use to save the generated code for later use
def write_to_file(file_name, string):
    my_file = open("generated_code/"+file_name+".txt", "w")
    my_file.write(string)
    my_file.close()
    

#string is executable c code
#x is the co-image of function
def Run_C(x, string):
    #function for power so as not to use #include<math.h>
    #power is one line so writing and reading to file is easier
    power  = "float power(float x, float n) {"
    power += "    float number = 1;"
    power += "    for (int i=0; i<n; ++i) {"
    power += "        number = number*x;"
    power += "    } return number;}"

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    
    x_dev = cl_array.to_device(queue, x)
    y_dev = cl_array.empty_like(x_dev)
    one = "__kernel void sum(__global float *x,"
    two = " __global float *y) {"
    code = power + one + two + string + "}"
    
    start = time.time()
    prg = cl.Program(ctx, code).build()
    print('Time to run C Code, ', time.time() - start)
    
    prg.sum(queue, (1,), None, x_dev.data, y_dev.data)
    return y_dev.get()


#run the C method from the given file using the given domain 
def Run_from_file(x, file_name):
    my_file = open("generated_code/"+file_name+".txt", "r")
    #code is just on first line of file, so get this then run it
    code = my_file.readline()
    out = Run_C(x, code)
    return out