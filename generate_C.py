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

#input is an approximator class
#output is C code.

#function for power so as not to use #include<math.h>
power = """
float power(float x, float n) {
    float number = 1;
    for (int i=0; i<n; ++i) {
        number = number*x;
    }
    return number;
}
"""

#first do simple thing
#make if statements for all ranges and evaluate them for just monomials
#doing this with a string.
def generate_srting(domain_size, ap):
    string = ""
    #string += "float power(float, float);  "
    #string += "int main() {"
    string += "float curr;"
    string += "for(int n=0; n<{0}; ++n)".format(int(domain_size))
    string += " { "
    for i in range(len(ap.ranges)):
        string += "curr = x[n];"
        string += "if (({0} <= curr) && (curr <= {1})) ".format(ap.ranges[i][0], ap.ranges[i][1])
        string += "{"
        string += "y[n] = "
        for j in range(ap.order):
            string += "{0}*power(curr, {1})".format(ap.coeff[i][j], j)
            if j != ap.order-1:
                string += " + "
        string += ";"
        string += "}"
    #string += "return y;"
    #string += "}"
    string += "}"        
    #string += power
    return string

#string is executable c code
#x is the co-image of function
def Run_C(x, string):    
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    
    x_dev = cl_array.to_device(queue, x)
    y_dev = cl_array.empty_like(x_dev)
    one = "__kernel void sum(__global float *x,"
    two = " __global float *y) {"
    code = power + one + two + string + "}"

    start = time.time()
    prg = cl.Program(ctx, code).build()
    print('Running C Code, ', time.time() - start)
    
    prg.sum(queue, (1,), None, x_dev.data, y_dev.data)
    return y_dev.get()


"""
x = np.linspace(0, 1, 10).astype(np.float32)

string = "for (int n=0; n<10; ++n){y[n] = power(x[n], 2);}"
y = Run_C(x, string)
print(y)
"""