"""
Script used to test the adaptive interpolation and
the evaluation of said interpolant
"""
from __future__ import absolute_import

import ctypes
import ctypes.util

import os
import time
import numpy as np
import numpy.linalg as la
import scipy.special as spec
import matplotlib as mpl
from tempfile import TemporaryDirectory
#import tempfile
#mpl.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import adaptive_interpolation.adapt as adapt
import adaptive_interpolation.approximator as app
import adaptive_interpolation.generate as generate
import adaptive_interpolation.adaptive_interpolation as adapt_i
#import loopy as lp
#from loopy.tools import (empty_aligned, address_from_numpy,
#        build_ispc_shared_lib, cptr_from_numpy)


def address_from_numpy(obj):
    ary_intf = getattr(obj, "__array_interface__", None)
    if ary_intf is None:
        raise RuntimeError("no array interface")

    buf_base, is_read_only = ary_intf["data"]
    return buf_base + ary_intf.get("offset", 0)


def cptr_from_numpy(obj):
    return ctypes.c_void_p(address_from_numpy(obj))


def build_ispc_shared_lib(
        cwd, ispc_sources, cxx_sources,
        ispc_options=[], cxx_options=[],
        ispc_bin="ispc",
        cxx_bin="g++",
        quiet=True):
    from os.path import join

    ispc_source_names = []
    for name, contents in ispc_sources:
        ispc_source_names.append(name)

        with open(join(cwd, name), "w") as srcf:
            srcf.write(contents)

    cxx_source_names = []
    for name, contents in cxx_sources:
        cxx_source_names.append(name)

        with open(join(cwd, name), "w") as srcf:
            srcf.write(contents)

    from subprocess import check_call

    ispc_cmd = ([ispc_bin,
                "--pic",
                "-o", "ispc.o"]
            + ispc_options
            + list(ispc_source_names))
    if not quiet:
        print(" ".join(ispc_cmd))

    check_call(ispc_cmd, cwd=cwd)

    cxx_cmd = ([
                cxx_bin,
                "-shared", "-Wl,--export-dynamic",
                "-fPIC",
                "-oshared.so",
                "ispc.o",
                ]
            + cxx_options
            + list(cxx_source_names))

    check_call(cxx_cmd, cwd=cwd)

    if not quiet:
        print(" ".join(cxx_cmd))



def build_scalar_shared_lib(
        cwd, cxx_sources,
        cxx_options=[],
        cxx_bin="g++",
        quiet=True):
    from os.path import join

    cxx_source_names = []
    for name, contents in cxx_sources:
        cxx_source_names.append(name)

        with open(join(cwd, name), "w") as srcf:
            srcf.write(contents)

    from subprocess import check_call

    cxx_cmd = ([cxx_bin,
                "-shared", "-Wl,--export-dynamic",
                "-fPIC",
                "-oshared.so",
                ]
            + cxx_options
            + list(cxx_source_names))
    check_call(cxx_cmd, cwd=cwd)
    if not quiet:
        print(" ".join(cxx_cmd))


# bessel function for testing
def f(x, order=0):
    return spec.jn(order, x)


def f0(x, v):
    if v == 0:
        return f(x)
    elif v == 1:
        return spec.jn(10, x)
    elif v== 2:
        return spec.hankel1(0, x)
    elif v == 3:
        return spec.hankel2(0, x)
    else:
        return spec.airy(x)



def make_code(size, order, precision, d, vectorized=True, approx=None, code=None, opt=[]):
    a, b = 0, 20
    if approx is None:
        approx = adapt_i.make_interpolant(a, b, f, order, precision,
                                          'chebyshev', dtype=d, optimizations=opt)
    else:
        approx.optimizations = opt
    if code is None:
        if vectorized:
            # vector width is actually set depending on data type.
            code = adapt_i.generate_code(approx, size=size, vector_width=8, cpu=True)            
        else:
            code = adapt_i.generate_code(approx, size=size, cpu=True)

    dt = approx.dtype_name
    var = "uniform " if vectorized else ""
    if vectorized:
        header = "export void eval("
    else:
        header = 'extern "C" void eval('
    if "arrays" in opt:
        header += "const " + var + dt + " mid[], "
        header += "const " + var + dt + " left[], "
        header += "const " + var + dt + " right[], "
        header += "const " + var + dt + " interval_a[], "
        header += "const " + var + dt + " interval_b[], "
        header += "const " + var + dt + " coeff[], "
    else:
        header += "const " + var + dt + " tree[], "


    if "map" in opt:
        fdt = "int64" if "calc intervals" in approx.optimizations else dt
        header += "const " + var + fdt + " f[], "

    header += var + dt + " x[], " + var + dt + " y[])"
    code = code.replace("\n", "\n\t")
    full_code = header + "{\n\n" + code + "\n}"
    return approx, full_code


def run_data(tree_depth, order, size, n, vec=True):
    if vec:
        flop = size*(4 + 2 + 2*(order-2))
    else:
        #flop = size*(5 + 3 + 5*(order-2))
        # with fused mult add / sub and if 2*x_scaled is done outside loop
        flop = size*(4 + 2 + 2*(order-2))
    memop = size*(4*tree_depth + order + 4)*4 # 4 bytes each access (single precision)



def run(approx, code, size, NRUNS, vec):
    if approx.dtype_name == "float":
        dt = np.float32
        STREAM_DTYPE = np.float32
        STREAM_CTYPE = ctypes.c_float

    elif approx.dtype_name == "double":
        dt = np.float64
        STREAM_DTYPE = np.float64
        STREAM_CTYPE = ctypes.c_double

    if 0:
        INDEX_DTYPE = np.int32
        INDEX_CTYPE = ctypes.c_int
    else:
        INDEX_DTYPE = np.int64
        INDEX_CTYPE = ctypes.c_longlong

    with open("tests/tasksys.cpp", "r") as ts_file:
        tasksys_source = ts_file.read()  

    with TemporaryDirectory() as tmpdir:
    #if 1:
        #tmpdir = os.getcwd() + "/gen"
        #print(tmpdir)
        #print(code)

        if "get asm" in approx.optimizations:
            if vec:
                with open(tmpdir+"/stream.ispc", 'w') as foo:
                    foo.writelines(code)
                mem_nam = "map" if "map" in approx.optimizations else "BST"
                if "test first loop" in approx.optimizations:
                    trav = "first_loop_vector"
                elif "test second loop" in approx.optimizations:
                    trav = "second_loop_vector"
                from subprocess import check_call
                ispc_cmd = (["/home/ubuntu-boot/Desktop/ispc-v1.9.1-linux/ispc", "stream.ispc",
                            "--pic", "-g", "--no-omit-frame-pointer",
                            "--target=avx2-i32x8",
                            "--arch=x86-64",
                            "--opt=disable-fma",
                            "--woff",
                            "-o", mem_nam+trav+".asm", "--emit-asm",])
                check_call(["ls"], cwd=tmpdir)                
                check_call(ispc_cmd, cwd=tmpdir)
            else:
                with open(tmpdir+"/eval.cpp", 'w') as foo:
                    foo.writelines(code)
                mem_nam = "map" if "map" in approx.optimizations else "BST"
                if "test first loop" in approx.optimizations:
                    trav = "first_loop_scalar"
                elif "test second loop" in approx.optimizations:
                    trav = "second_loop_scalar"
                from subprocess import check_call
                scalar_cmd = (["g++",
                            "-g", "-fopenmp", 
                            "-DISPC_USE_OMP", "-std=c++11", "-march=broadwell",
                            "-S", "-o", mem_nam+trav+".s", "eval.cpp"])
                check_call(scalar_cmd, cwd=tmpdir)
            return
        else:
            # -march g++ cpu flag causes vectorization of scalar code, but this
            # is the family that the cpu is so will it be auto vectorized anyways on dunkel?
            # when running the compilar on my own it seems like it isnt..
            if vec:
                build_ispc_shared_lib(
                        tmpdir,
                        [("stream.ispc", code)],
                        [("tasksys.cpp", tasksys_source)],
                        cxx_options=["-g", "-O0", "-fopenmp", "-DISPC_USE_OMP", "-std=c++11"],
                        ispc_options=([
                            "-g", "-O0", "--no-omit-frame-pointer",
                            "--arch=x86-64",
                            #"--opt=force-aligned-memory",
                            #"--opt=fast-math",
                            "--opt=disable-fma",
                            "--woff",
                            ]
                            # if not vectorized then compile as scalar c++ code.
                            + (["--target=avx2-i32x8"] if vec else [])
                            #+ (["--opt=disable-loop-unroll"] if "unroll" in approx.optimizations 
                            #                                 or "unroll_order" in approx.optimizations else [])
                            + (["--addressing=64"] if INDEX_DTYPE == np.int64 else [])
                            ),
                        ispc_bin="/home/ubuntu-boot/Desktop/ispc-v1.9.1-linux/ispc",
                        )
            else:
                build_scalar_shared_lib(
                        tmpdir,
                        [("eval.cpp", code)],
                        cxx_options=[
                            "-g", "-O0", "-fopenmp", "-DISPC_USE_OMP", "-std=c++11"]
                        )
                os.system("objdump -S "+tmpdir+"/shared.so > asm.s")
                with open("asm.s", 'r') as foo:
                    string = foo.readlines()
                    if ("vmul" in string) or ("vadd" in string):
                        print("SCALAR CODE IS VECTORIZED!")
        x = np.linspace(approx.lower_bound,
                        approx.upper_bound, 
                        size,
                        endpoint=False,
                        dtype=dt)
        if "random" in approx.optimizations:
            np.random.shuffle(x)
        y = np.zeros(size, dtype=dt)
        approx.tree_1d = np.array(approx.tree_1d, dtype=dt)
        approx.mid = np.array(approx.mid, dtype=dt)
        approx.left = np.array(approx.left, dtype=dt)
        approx.right = np.array(approx.right, dtype=dt)
        approx.interval_a = np.array(approx.interval_a, dtype=dt)
        approx.interval_b = np.array(approx.interval_b, dtype=dt)
        approx.coeff  = np.array(approx.coeff, dtype=dt)


        knl_lib = ctypes.cdll.LoadLibrary(os.path.join(tmpdir, "shared.so"))

        xn = x[0]
        index = approx.map[int((xn - 0.0)*1.6)]
        l = ((np.int64(index) >> 20) & 0xfffff)
        r = ((np.int64(index) >> 40) & 0xfffff)
        print(bin(np.int64(index)))
        index = np.int64(index) & 0xfffff
        a = 0.625 * l + 0.0
        b = 0.625 * r + 0.0
        print(index)
        print(xn, l, r, a, b, len(approx.coeff))
        x_scaled = (2./(b - a))*(xn - a) - 1.0
        index = index*3
        print(x_scaled)
        print(index  , approx.coeff[index])
        print(index+1, approx.coeff[index+1])
        print(index+2, approx.coeff[index+2])


        def call_kernel():
            if 'map' in approx.optimizations:

                knl_lib.eval(
                        cptr_from_numpy(approx.mid),
                        cptr_from_numpy(approx.left),
                        cptr_from_numpy(approx.right),
                        cptr_from_numpy(approx.interval_a),
                        cptr_from_numpy(approx.interval_b),
                        cptr_from_numpy(approx.coeff),
                        cptr_from_numpy(approx.map),
                        cptr_from_numpy(x),
                        cptr_from_numpy(y),
                        INDEX_CTYPE(size),
                        )
            else:
                knl_lib.eval(
                        cptr_from_numpy(approx.tree_1d),
                        cptr_from_numpy(x),
                        cptr_from_numpy(y),
                        INDEX_CTYPE(size),
                        )

        # clear the kernel
        for i in range(8):
            call_kernel()

        #plt.figure()
        #plt.plot(x, y)
        #plt.plot(x, abs(y - f(x)))
        #plt.show()

        start_time = time.time()
        for irun in range(NRUNS):
            call_kernel()
        elapsed = time.time() - start_time

        FLOPS = (4 + 2 + 2*(approx.max_order-2))
        print("Average Runtime:", elapsed/NRUNS)
        nbytes = 4 if approx.dtype_name == 'float' else 8
        if 'map' in approx.optimizations:
            memop = size*(1 + (approx.max_order-1) + 3)*nbytes
        else:
            memop = size*(3*approx.num_levels + (approx.max_order-1) + 3)*nbytes
        # times size*4 because thats the number of bytes in x
        GFLOPS = FLOPS*size/(2**30)
        #print(GFLOPS/elapsed, "GFLOPS/s")
        z = f(x[::2048])
        plt.figure()
        print(y)
        #plt.yscale("log")
        #plt.scatter(x, np.abs(y-f(x)))
        #plt.show()
        print("Relative Error:", la.norm(z-y[::2048], np.inf)/la.norm(z, np.inf))
        return y


def run_multiple(orders, precisions, opt=[]):
    # 2**30 is a GigaByte
    # 2**40 is a TerraByte
    # uh oh, it seems like half, 2.84 seconds is taken in the c++ script...
    # yup, i get a 2x speed up when building the library
    # guess i need to make a shared library
    print()    
    print(opt)
    num_samples = 15
    size = 2**26#8*2**26
    with_scalar = True
    dt = '64' if '64' in opt else '32'
    npdt = np.float64 if '64' in opt else np.float32
    for precision in precisions:
        for order in orders:
            print("Vector: ", order, precision)
            approx, ispc_code = make_code(size, order, precision, dt, vectorized=True, opt=opt)
            #print("tree levels: ", approx.num_levels)
            y = run(approx, ispc_code, size, num_samples, True)

            if with_scalar:
                print("Scalar: ", order, precision)
                approx, ispc_code = make_code(size, order, precision, dt, vectorized=False, approx=approx, opt=opt)
                y = run(approx, ispc_code, size, num_samples, False)

    if 0:
        plt.figure()
        plt.title("GFLOPS vs. Order, precision= "+repr(precision))
        # max is 46.4
        plt.plot(orders, np.array(orders)*0 + 35.2, label="Max Vectorized GFLOPS (Dunkel)")
        plt.errorbar(orders, gflops_vect, yerr=gflops_vect_err, label="Vectorized")
        if with_scalar:
            # max is 5.8
            plt.plot(orders, np.array(orders)*0 + 4.4, label="Max Scalar GFLOPS (Dunkel)")
            plt.errorbar(orders, gflops_scalar, yerr=gflops_scalar_err, label="Scalar")
            plt.legend()
        plt.savefig("figs/"+repr(time.time())+"GFLOPs"+repr(precision)+".png")

        plt.figure()
        plt.title("Memory Bandwidth vs. Order, precision= "+repr(precision))
        plt.plot(orders, np.array(orders)*0 + 76.8, label="Max Memory Bandwidth (Dunkel)")
        plt.errorbar(orders, mb_vect, yerr=mb_vect_err, label="Vectorized")
        if with_scalar:
            plt.errorbar(orders, mb_scalar, yerr=mb_scalar_err, label="Scalar")
        plt.legend()
        plt.savefig("figs/"+repr(time.time()%60)+"MB"+repr(precision)+".png")


def run_one(approx, precision, opt=[]):
    print()
    print(opt)
    num_samples = 15
    size = 2**12
    order = approx.max_order
    dt = '64' if '64' in opt else '32'
    #print("Vector: ", order, precision)
    approx, ispc_code = make_code(size, order, precision, dt, vectorized=True, opt=opt, approx=approx)
    print(ispc_code)
    y = run(approx, ispc_code, size, num_samples, True)

    if 'prefetch' not in opt:
        #print("Scalar: ", order)
        approx, ispc_code = make_code(size, order, precision, dt, vectorized=False, approx=approx, opt=opt)
        ispc_code = ispc_code.replace("int64", "long long")
        print(ispc_code)
        y = run(approx, ispc_code, size, num_samples, False)


def test():
    # Function used to obtain results. DONT CHANGE
    a, b = 0, 20
    order, precision = 5, 1e-4

    approx = adapt_i.make_interpolant(a, b, f, order, precision, 'chebyshev', dtype=32, optimizations=["arrays"])

    run_one(approx, precision, opt=[])
    #run_one(approx, precision, opt=['unroll'])
    #run_one(approx, precision, opt=['unroll_order'])
    #run_one(approx, precision, opt=['prefetch'])
    #run_one(approx, precision, opt=['unroll', 'unroll_order', 'prefetch'])
    run_one(approx, precision, opt=['test first loop'])
    run_one(approx, precision, opt=['test second loop'])

    run_one(approx, precision, opt=['arrays', 'map'])
    #run_one(approx, precision, opt=['arrays', 'map', 'unroll'])
    #run_one(approx, precision, opt=['arrays', 'map', 'unroll_order'])
    #run_one(approx, precision, opt=['arrays', 'map', 'prefetch'])
    #run_one(approx, precision, opt=['arrays', 'map', 'unroll', 'unroll_order', 'prefetch'])
    run_one(approx, precision, opt=['arrays', 'map', 'test first loop'])
    run_one(approx, precision, opt=['arrays', 'map', 'test second loop'])

    #run_one(approx, precision, opt=['none'])

    print("USING DOUBLES")
    approx = adapt_i.make_interpolant(a, b, f, order, precision, 'chebyshev', dtype=64, optimizations=["arrays"])

    run_one(approx, precision, opt=['64'])
    #run_one(approx, precision, opt=['64', 'unroll'])
    #run_one(approx, precision, opt=['64', 'unroll_order'])
    #run_one(approx, precision, opt=['64', 'prefetch'])
    #run_one(approx, precision, opt=['64', 'unroll', 'unroll_order', 'prefetch'])
    run_one(approx, precision, opt=['64', 'test first loop'])
    run_one(approx, precision, opt=['64', 'test second loop'])

    run_one(approx, precision, opt=['64', 'arrays', 'map'])
    #run_one(approx, precision, opt=['64', 'arrays', 'map', 'unroll'])
    #run_one(approx, precision, opt=['64', 'arrays', 'map', 'unroll_order'])
    #run_one(approx, precision, opt=['64', 'arrays', 'map', 'prefetch'])
    #run_one(approx, precision, opt=['64', 'arrays', 'map', 'unroll', 'unroll_order', 'prefetch'])
    run_one(approx, precision, opt=['64', 'arrays', 'map', 'test first loop'])
    run_one(approx, precision, opt=['64', 'arrays', 'map', 'test second loop'])

    #run_one(approx, opt=['64', 'none'])

    precision = 1e-13
    approx = adapt_i.make_interpolant(a, b, f, order, precision, 'chebyshev', dtype=64, optimizations=["arrays"])
    run_one(approx, precision, opt=['64'])
    run_one(approx, precision, opt=['64', 'test first loop'])
    run_one(approx, precision, opt=['64', 'test second loop'])

    run_one(approx, precision, opt=['64', 'arrays', 'map'])
    run_one(approx, precision, opt=['64', 'arrays', 'map', 'test first loop'])
    run_one(approx, precision, opt=['64', 'arrays', 'map', 'test second loop'])

    """
    precisions32 = [1e-7]
    precisions64 = [1e-7, 1e-13]
    orders = [5, 9, 14]
    run_multiple(orders, precisions32, opt=[])
    run_multiple(orders, precisions32, opt=['arrays', 'map'])
    run_multiple(orders, precisions64, opt=['64'])
    run_multiple(orders, precisions64, opt=['64', 'arrays', 'map'])
    """


def get_asm():
    a, b = 0, 20
    order, precision = 5, 1e-6
    print("make approximation")
    approx = adapt_i.make_interpolant(a, b, f, order, precision, 'chebyshev', dtype=32, optimizations=["arrays"])
    print("Run approximation")

    run_one(approx, opt=['test first loop', 'get asm'])
    run_one(approx, opt=['test second loop', 'get asm'])

    run_one(approx, opt=['arrays', 'map', 'test first loop', 'get asm'])
    run_one(approx, opt=['arrays', 'map', 'test second loop', 'get asm'])


def new_test():
    # Function used to obtain results. DONT CHANGE
    a, b = 0, 20
    order, precision = 5, 1e-6

    approx = adapt_i.make_interpolant(a, b, f, order, precision, 'chebyshev', dtype=32, optimizations=["arrays"])

    run_one(approx, precision, opt=['test flops'])
    run_one(approx, precision, opt=['none'])

    run_one(approx, precision, opt=['arrays', 'map'])
    run_one(approx, precision, opt=['arrays', 'map', 'test first loop'])
    run_one(approx, precision, opt=['arrays', 'map', 'test second loop'])

    run_one(approx, precision, opt=[])
    run_one(approx, precision, opt=['test first loop'])
    run_one(approx, precision, opt=['test second loop'])


    print("Doubles: Same Precision")
    approx = adapt_i.make_interpolant(a, b, f, order, precision, 'chebyshev', dtype=64, optimizations=["arrays"])

    # give peak flop estimate
    run_one(approx, precision, opt=['64', 'test flops'])
    run_one(approx, precision, opt=['64', 'none'])

    run_one(approx, precision, opt=['64', 'arrays', 'map'])
    run_one(approx, precision, opt=['64', 'arrays', 'map', 'test first loop'])
    run_one(approx, precision, opt=['64', 'arrays', 'map', 'test second loop'])

    run_one(approx, precision, opt=['64'])
    run_one(approx, precision, opt=['64', 'test first loop'])
    run_one(approx, precision, opt=['64', 'test second loop'])


    print("Doubles: Higher Precision")
    order, precision = 5, 1e-13
    approx = adapt_i.make_interpolant(a, b, f, order, precision, 'chebyshev', dtype=64, optimizations=["arrays"])

    # give peak flop estimate
    run_one(approx, precision, opt=['64', 'test flops'])
    run_one(approx, precision, opt=['64', 'none'])

    run_one(approx, precision, opt=['64', 'arrays', 'map'])
    run_one(approx, precision, opt=['64', 'arrays', 'map', 'test first loop'])
    run_one(approx, precision, opt=['64', 'arrays', 'map', 'test second loop'])

    run_one(approx, precision, opt=['64'])
    run_one(approx, precision, opt=['64', 'test first loop'])
    run_one(approx, precision, opt=['64', 'test second loop'])




# run the main program
if __name__ == "__main__":
    #get_asm()
    #new_test()
    # Function used to obtain results. DONT CHANGE
    a, b = 0, 20
    order, precision = 2, 1e-3
    opt = ["arrays", "map", "calc intervals", "random"]
    approx = adapt_i.make_interpolant(a, b, f, order, 
                                      precision, 'chebyshev', 
                                      dtype=32, optimizations=opt)
    run_one(approx, precision, opt=opt)
    
