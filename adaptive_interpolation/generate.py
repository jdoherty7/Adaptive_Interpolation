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
    # maximum possible order of representation
    order = int(ap.max_order)
    if single:
        string  = "for (int n=get_global_id(0); n<" + repr(int(ap.size))
        string += "; n+=" + repr(int(ap.vector_width)) + ") {\n"
    else:
        string  = "int n = get_global_id(0);\n"
    # gives the index of the coefficients to use
    string += "int index = 1;\n"
    string += ap.dtype + " x_const = x[n];\n"
    string += "for (int i=1; i<{0}; i++)".format(int(ap.num_levels))
    string += "{\n\tindex = tree[index] > x[n] ? "
    string += "(int)tree[index+{0}] : ".format(4+order)
    string += "(int)tree[index+{0}];\n".format(5+order)
    string += "}\n"
    string += "y[n] = "
    # the coefficients are transformed from a matrix to a vector.
    # the formula to call the correct entry is given as the indices
    sub_string = "tree[index+{0}]".format(order+2)
    for j in range(order)[::-1]:
        # using horner's method, this requires the for loop to be reversed
        sub_string = "x_const*(" + sub_string + \
                     ") + tree[index+{0}]".format(j + 1)
    string += "" + sub_string
    string += ";\n"
    if single:
        string+= "}"
    return string


# generate C code that evaluates chebyshev polynomials
# according to the approximator class that is given.
def gen_cheb(ap):
    single = False if (ap.size is None or ap.vector_width is None) else True
    # maximum possible order of representation
    order = int(ap.max_order)

    if single:
        string  = "for (int n=get_global_id(0); n<" + repr(int(ap.size))
        string += "; n+=" + repr(int(ap.vector_width)) + ") {\n"
    else:
        string  = "int n = get_global_id(0);\n"
    # gives the index of the coefficients to use
    string += "int index = 0;\n"
    string += ap.dtype + " T0, T1, Tn, a, b, s, x_scaled;\n"
    string += "for (int i=1; i<{0}; i++)".format(int(ap.num_levels))
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
        string  = "for (int n=get_global_id(0); n<" + repr(int(ap.size))
        string += "; n+=" + repr(int(ap.vector_width)) + ") {\n"
    else:
        string  = "int n = get_global_id(0);\n" 
    # gives the index of the coefficients to use
    string += "int index = 0;\n"
    string += ap.dtype + " L0, L1, Ln, a, b, s, x_scaled;\n"
    string += "for (int i=1; i<{0}; i++)".format(int(ap.num_levels))
    string += "{\n\tindex = tree[index] > x[n] ? "
    string += "(int)tree[index+{0}] : ".format(4+order)
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
        dt = approx.dtype
        declaration = "__kernel void sum(__global " + dt + "  *tree, "
        declaration += "__global "+ dt +" *x, __global "+ dt +" *y) "
        code = declaration + '{' + approx.code + '}'

        # compile code and then execute it
        prg = cl.Program(ctx, code).build()

        prg.sum(queue, x_dev.shape, None, tree_dev.data, 
                x_dev.data, y_dev.data)

        queue.finish()

        return y_dev.get()
    else:
        raise ValueError("Function requires pyopencl installation.")


# generates ispc code that can be run and compiled with ispc 
def gen_ispc(ap):
    if "none" in  ap.optimizations:
        return "y[0] = x[0]+10;"

    order = int(ap.max_order)
    # generate vectorized or scalar code.
    vec = False
    if ap.vector_width is not None:
        if ap.vector_width > 1:
            vec=True
    
    if "scalar" in ap.optimizations:
        vec = False

    if "arrays" in ap.optimizations or "map" in ap.optimizations:
        interval_a_index, interval_b_index, left_index, right_index, coeff_index = ['']*5
        # change from two arrays for a and b to one
        interval_a_index, interval_b_index = '', 1
    elif "trim_data" in ap.optimizations:
        interval_a_index = order+4
        interval_b_index = order+5
        left_index = 1
        right_index = 2
        coeff_index = 3
    else:
        interval_a_index = 2+order
        interval_b_index = 3+order
        left_index = 4+order
        right_index = 5+order
        coeff_index = 1

    if "arrays" in ap.optimizations or "map" in ap.optimizations:
        left = "left"
        right = "right"
        mid = "mid"
        # using two arrays for a and b
        #interval_a = "interval_a"
        #interval_b = "interval_b"
        interval_a = "intervals"
        interval_b = "intervals"
        coeff = "coeff"
    else:
        # set all variables to the string "tree"
        left, right, mid, interval_a, interval_b, coeff = ["tree"]*6



    vtype = "varying " if vec else "uniform "
    string = "\n"
    for var in ["T0", "T1", "Tn", "a", "b", "s", "x_scaled", "xn"]:
        string += vtype + ap.dtype_name + " " +var + " = 0;\n"
    if "output" not in ap.optimizations:
        string += vtype + ap.dtype_name + " y = 0;\n"


    if vec:
        string += "for (uniform int nbase=0; nbase<" + repr(int(ap.size))
        string += "; nbase+=programCount) {\n\n"
        string += "varying int n = nbase + programIndex;\n"
    else:
        string += "for (uniform int nbase=0; nbase<" + repr(int(ap.size))
        string += "; nbase++) {\n\n"
        string += vtype + "int n = nbase;\n"

    if "test flops" in ap.optimizations:
        #string += "a = x[n]*79.89;\n"
        string += "s = x[n]*6.4;\n}"
        return string

    if "prefetch" in ap.optimizations:
        if "arrays" in ap.optimizations:
            string += "prefetch_l1(&mid[{0}]);\n".format(len(ap.mid)-1)
            string += "prefetch_l1(&right[{0}]);\n".format(len(ap.right)-1)
            string += "prefetch_l1(&left[{0}]);\n".format(len(ap.left)-1)
            string += "prefetch_l1(&coeff[{0}]);\n".format(len(ap.coeff)-1)
        else:
            string += "prefetch_l1(&tree[{0}]);\n".format(ap.tree.size)

    # gives the index of the coefficients to use
    # long long is int64 in ispc
    string += vtype + "int index = 0;\n"

    if "calc intervals" in ap.optimizations:
        string += vtype + "int l = 0;\n"
        string += vtype + "int L = 0;\n"
        ty = "int" if (2*ap.D + ap.lgD) < 32 else "int64"
        string += vtype + ty + " data = 0;\n"

    
    if "output" in ap.optimizations:
        string += "xn = x[n];\n"
    else:
        # equispaced points in [a, b)
        string += "xn = {0} + n*{1:.90f};\n".format(ap.lower_bound, (ap.upper_bound - ap.lower_bound-.01)/(ap.size+1))


    if "test second loop" in ap.optimizations:
        intro = string

    if "map" in ap.optimizations:
        # N is the size of the mapping function
        #N = 2**(ap.num_levels-1)
        N = len(ap.map)
        # you do want to scale to N, but b will equal N which is too large of an index
        # therefore this method cannot inclued the endpoint, b
        # int( N*(x-a)/b) rescale from [a, b] to [0, N] then get index
        if "calc intervals" in ap.optimizations:
            string += "data = "
        else:
            string += "index = "
        string += "f[(int)({0}*xn - {1})];\n".format(N/(ap.upper_bound - ap.lower_bound), 
                                            ap.lower_bound*N/(ap.upper_bound - ap.lower_bound) )
        
    else:
        if "unroll" in ap.optimizations:
            for i in range(1, int(ap.num_levels)):
                string += "index = {0}[index] > xn ? ".format(mid)
                string += "(int){0}[index+{1}] : ".format(left, left_index)
                string += "(int){0}[index+{1}];\n".format(right, right_index)
        else:
            string += "for (" + vtype + "int i=1; i<{0}; i++)".format(int(ap.num_levels))
            string += "{\n"
            string += "\tindex = {0}[index] > xn ? ".format(mid)
            string += "(int){0}[index+{1}] : ".format(left, left_index)
            string += "(int){0}[index+{1}];\n".format(right, right_index)
            string += "} \n"

    # used for testing performance of JUST the first or second loop
    # by removing the other loop from the generated code
    if "test first loop" in ap.optimizations:
        if "output" in ap.optimizations:
            string += "y[n] = s;\n}\n"
        else:
            string += "y = s;\n}\n"
            #string += "uniform "+ap.dtype_name + " ret = extract(y, 0);\n"
            string += "ret[0] = extract(y, 0);\n"
        string = string.replace("varying int i=", "uniform int i=")
        string = string.replace("varying int j=", "uniform int j=")
        return string
    
    if "test second loop" in ap.optimizations:
        string = ""

    # rescale variables
    if "calc intervals" in ap.optimizations:
        string += "l = (data >> {0}) & {1};\n".format(ap.num_levels, hex(ap.D_ones))
        string += "L = (data >> {0});// & {1};\n".format(2*ap.num_levels, hex(ap.lgD_ones))
        string += "index = data & {0};\n".format(hex(ap.D_ones))
        scaling = (ap.upper_bound -  ap.lower_bound) / len(ap.map)
        
        op = "+" if ap.lower_bound >=0 else "-"
        string += "a = {0} * ({1})l {2} {3};\n".format(scaling, ap.dtype_name, op, abs(ap.lower_bound))
        string += "b = {0} * ({1})(l+L) {2} {3};\n".format(scaling, ap.dtype_name, op, abs(ap.lower_bound))
        # this works, but id like to do the division while generating. probably can't because
        # the operations are not commutative. Or it might be because L isnt actually converted
        # to double before dividing even though I do the cast...
        string += "x_scaled = (2./(b-a))*(xn - a) - 1.0;\n".format(scaling, ap.dtype_name)
        #string += "x_scaled = (2./({0}*({1})L))*(xn - a) - 1.0;\n".format(scaling, ap.dtype_name)

    else:
        string += "a = {0}[index+{1}];\n".format(interval_a, interval_a_index)
        string += "b = {0}[index+{1}];\n".format(interval_b, interval_b_index)
        string += "x_scaled = (2./(b - a))*(xn - a) - 1.0;\n"
    #string += "x_scaled = 2*((xn-a)/(b - a)) - 1.0;\n"
    # which of the above is better?
    if "arrays" in ap.optimizations:
        string += "index = index*{0};\n".format(order+1)
    string += "\nT0 = 1.0;\n"
    # initialize the sum
    string += "s = {0}[index+{1}]*T0;\n".format(coeff, coeff_index)
    if "arrays" in ap.optimizations:
        string = string.replace("index+]", "index]")
        #string = string.replace("index+", "index")
    coeff_index = 0 if coeff_index == '' else coeff_index
    if order > 0:
        string += "T1 = x_scaled;\n"
        string += "s = s + {0}[index+{1}]*T1;\n".format(coeff, coeff_index+1)
    if order > 1:
        # rescale x for the loop
        string+= "x_scaled = 2*x_scaled;\n"
        if "unroll_order" in ap.optimizations and order < 5000:
            string += '\n'
            for j in range(2+coeff_index, order+1+coeff_index):
                string += "Tn = x_scaled*T1 - T0;\n"
                string += "s = s + {0}[index + ".format(coeff) +repr(j)+"]*Tn;\n"
                string += "T0 = T1;\n"
                string += "T1 = Tn;\n"
        else:
            # calculate order 2 through n
            #loop_type = "int64" if "calc intervals" in ap.optimizations else "int"
            string += "for (" + vtype+"int j="+repr(coeff_index+2)+"; j <="
            string += repr(order+coeff_index) + "; j++) {\n"
            string += "\tTn = x_scaled*T1 - T0;\n"
            string += "\ts = s + {0}[index + j]*Tn;\n".format(coeff)
            string += "\tT0 = T1;\n"
            string += "\tT1 = Tn;\n"
            string += "}\n"

    if "output" in ap.optimizations:
        string += "y[n] = s;\n}\n"
    else:
        string += "y = y + s;\n}\n"
        #string += "uniform "+ap.dtype_name + " ret = extract(y, 0);\n"
        string += "ret[0] = extract(y, 0);\n"

    if "test second loop" in ap.optimizations:
        string = intro + string
    string = string.replace("varying int i=", "uniform int i=")
    string = string.replace("varying int j=", "uniform int j=")
    return string



# builds a pyopencl kernel object that can be used to 
# run the generated and built code multiple times with different inputs
# if ispc given generate ispc code and return as string.
# TODO: rename to build kernel
def build_code(approx, ispc=False):
    if ispc:
        vtype = ""
        if approx.code is None:
            raise Exception("Need to generate code before building the kernel.")


        dt = approx.dtype_name
        # removed if generating C++
        vtype = "uniform "
        header = "export void eval("    
        # used if generating code for c++ 
        #header = 'extern "C" void eval('
        if "calc intervals" not in approx.optimizations:
            header += "const " + vtype + dt + " intervals[], "
            #header += "const " + vtype + dt + " interval_a[], "
            #header += "const " + vtype + dt + " interval_b[], "
        
        header += "const " + vtype + dt + " coeff[], "
        if "map" in approx.optimizations:

            if "calc intervals" in approx.optimizations:
                if (2*approx.D + approx.lgD) < 32:
                    fdt = "int"
                else:
                    fdt = "int64"
            else:
                fdt = "int"
            header += "const " + vtype + fdt + " f[]"
    
        if "output" in approx.optimizations:
            header += ", const "+vtype + dt + " x[], " + vtype + dt + " y[]"
        else:
            header += ", "+vtype + dt + " ret[]"


        code = approx.code.replace("\n", "\n\t")
        kernal = header + "){\n\n" + code + "\n}"

        return kernal

    if with_pyopencl:
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        queue.finish()
        tree = approx.tree_1d
        tree_dev = cl_array.to_device(queue, tree)

        # build the code to run from given string
        dt = approx.dtype_name
        declaration = "__kernal void eval(__global " + dt + "  *tree, "
        declaration += "__global " + dt + " *x, __global " + dt + " *y) "
        code = declaration + '{' + approx.code + '}'
        
        prg = cl.Program(ctx, code).build()
        #print()
        print(ctx.devices[0])
        #print("\n\n\n\n", prg.binaries[0], "\n\n\n\n\n\n")

        knl = prg.eval
        queue.finish()

        approx.kernal   = knl
        approx.queue    = queue
        approx.tree_dev = tree_dev

        return knl, queue, tree_dev
    else:
        raise ValueError("Function requires pyopencl installation.")


def run_single(x, ap):
    x_dev = cl_array.to_device(ap.queue, x)
    y_dev = cl_array.empty_like(x_dev)

    ap.queue.finish()
    start = time.time()
    ap.kernal(ap.queue, (int(ap.vector_width),), (int(ap.vector_width),),
              ap.tree_dev.data, x_dev.data, y_dev.data)
    ap.queue.finish()
    end = time.time() - start
    return end, y_dev.get()


def run_multi(x, ap):
    x_dev = cl_array.to_device(ap.queue, x)
    y_dev = cl_array.empty_like(x_dev)

    ap.queue.finish()
    start = time.time()
    ap.knl(ap.queue, x_dev.shape, None, ap.tree_dev.data, x_dev.data, ay_dev.data)
    ap.queue.finish()
    end = time.time() - start
    return end, y_dev.get()


