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
        assert approx.dtype == np.float32
        STREAM_DTYPE = np.float32
        STREAM_CTYPE = ctypes.c_float
    elif approx.dtype_name == "double":
        assert approx.dtype == np.float64
        STREAM_DTYPE = np.float64
        STREAM_CTYPE = ctypes.c_double


    if "calc intervals" in approx.optimizations:
        INDEX_DTYPE = np.int64
        INDEX_CTYPE = ctypes.c_longlong
    else:
        INDEX_DTYPE = np.int32
        INDEX_CTYPE = ctypes.c_int




    with open("tests/tasksys.cpp", "r") as ts_file:
        tasksys_source = ts_file.read()  

    with TemporaryDirectory() as tmpdir:
    #if 1:
        #tmpdir = os.getcwd() + "/gen"
        #print(tmpdir)
        #print(code)

        # -march g++ cpu flag causes vectorization of scalar code, but this
        # is the family that the cpu is so will it be auto vectorized anyways on dunkel?
        # when running the compilar on my own it seems like it isnt..
        build_ispc_shared_lib(
                tmpdir,
                [("stream.ispc", code)],
                [("tasksys.cpp", tasksys_source)],
                cxx_options=[
                    #"-g", "-O0",
                    "-fopenmp", "-DISPC_USE_OMP", "-std=c++11"],
                ispc_options=([
                    # -g causes optimizations to be disabled
                    # -O0 turns off default optimizations (three levels available)
                    "-g", "-O1", "--no-omit-frame-pointer",
                    "--arch=x86-64",
                    #"--opt=force-aligned-memory",
                    #"--opt=fast-math",
                    #"--opt=disable-fma",
                    # turn off error messaging
                    "--woff",
                    #"--opt=disable-loop-unroll",
                    "--cpu=core-avx2",
                    "--target=avx2-i32x8",
                    ]
                    #+ (["--opt=disable-loop-unroll"] if "unroll" in approx.optimizations 
                    #                                 or "unroll_order" in approx.optimizations else [])
                    # this is needed because map is int64 ?
                    # only need to use if accessing more than 4 GB of information?
                    + (["--addressing=32"])
                    ),
                ispc_bin= "/home/ubuntu-boot/Desktop/ispc-v1.9.1-linux/ispc",
                )

        dt = approx.dtype
        x = np.linspace(approx.lower_bound,
                        #1.1,
                        approx.upper_bound, 
                        size,
                        endpoint=False,
                        dtype=dt)
        if "random" in approx.optimizations:
            np.random.shuffle(x)

        # make sure that these are already numpy arrays of the correct type..
        y = np.zeros(size, dtype=dt)
        approx.tree_1d    = np.array(approx.tree_1d, dtype=dt)
        approx.interval_a = np.array(approx.interval_a, dtype=dt)
        approx.interval_b = np.array(approx.interval_b, dtype=dt)
        approx.coeff      = np.array(approx.coeff, dtype=dt)



        knl_lib = ctypes.cdll.LoadLibrary(os.path.join(tmpdir, "shared.so"))
        g = knl_lib.eval
        if 'map' in approx.optimizations:
            if "calc intervals" in approx.optimizations:
                args = [cptr_from_numpy(approx.coeff),
                        cptr_from_numpy(approx.cmap),
                        cptr_from_numpy(x),
                        cptr_from_numpy(y),
                    ]
            else:
                args = [cptr_from_numpy(approx.interval_a),
                        cptr_from_numpy(approx.interval_b),
                        cptr_from_numpy(approx.coeff),
                        cptr_from_numpy(approx.map),
                        cptr_from_numpy(x),
                        cptr_from_numpy(y),
                        ]
        else:
            # evaluating using BST for interval search
            args = [cptr_from_numpy(approx.tree_1d),
                    cptr_from_numpy(x),
                    cptr_from_numpy(y),
                    ]
    
        # run before instantiating too??
        for i in range(2):
            g(*args)

        def call_kernel():
            g(*args)

        # clear the kernel
        for i in range(30):
            call_kernel()

        if "graph" in approx.optimizations:
            s = 2048
            if 0:
                plt.figure()
                plt.title("Function")
                plt.scatter(x[::s], y[::s])
                plt.show()
            if 1:
                plt.figure()
                plt.title("Absolute Error")
                plt.yscale("log")
                plt.plot(x[::s], abs(y[::s] - f(x[::s])))
                plt.show()

        start_time = time.time()
        for _ in range(NRUNS):
            call_kernel()
        elapsed = time.time() - start_time

        # Automatically calculate Memory Bandwidth and GigaFlops.
        #FLOPS = (4 + 2 + 2*(approx.max_order-2))
        # reduction + scale + first terms + order loop
        nbytes = 4 if approx.dtype_name == 'float' else 8
        d = 4 if approx.cmap.dtype == np.int32 else 8
        if "calc intervals" in approx.optimizations:
            # without the interval storage
            FLOPS  =  2 + 2 + 5 + 1 + 4*approx.max_order
            memops = (3 +   approx.max_order)*nbytes + d
        else:
            # with the interval storage
            FLOPS  =  2 + 5 + 1 + 4*approx.max_order
            memops = (5 +   approx.max_order)*nbytes + d

        # mem reciprocal throughput of instruction between 7 and 12
        print("Average Runtime (ns) per x:", (1e9)*elapsed/NRUNS/size)
        # times size*4 because thats the number of bytes in x
        avgtime = elapsed/NRUNS
        GFLOPS = (FLOPS/avgtime )*(size/(10**9))#(2**30)
        MEMBND = (memops/avgtime)*(size/(10**9))

        vw = 8 if approx.dtype_name == "float" else 4 # for double, non-turbo
        peakGF = 8.8*vw
        peakMB = 10.88#76.8

        #print("Flops/Byte: ", (FLOPS/avgtime)/(memops*size))
        print("GFLOPS/s:   ", GFLOPS, " (Max = "+str(peakGF)+") ", GFLOPS/peakGF)
        print("MB (GB/s):  ", MEMBND, " (Max = "+str(peakMB)+" GB/s) ", MEMBND/peakMB)
        print("Total Use:  ", (GFLOPS/peakGF) + (MEMBND/peakMB))
        s = 2048
        z = f(x[::s])
        a = la.norm(z-y[::s], np.inf)
        r = a/la.norm(z, np.inf)
        #if r > approx.allowed_error:
        print("Relative Error:", r)
        print("Absolute Error:", a)

        return GFLOPS, avgtime/size




def run_one(approx, size, num_samples, opt=[]):
    print()
    print(opt)
    #print("Vector: ", order, precision)
    approx.optimizations = opt
    pre_header_code = adapt_i.generate_code(approx, size=size, vector_width=8, cpu=True)
    ispc_code       = generate.build_code(approx, ispc=True)

    # Bytes of floating point type used
    #######################################################
    f = 4 if approx.dtype_name == "float" else 8
    L, s = approx.leaf_index + 1, len(approx.map)
    d = 4 if approx.cmap.dtype == np.float32 else 8
    if "calc intervals" in approx.optimizations:
        STORAGE = (s*(d/f) + 2*size + approx.max_order*L)*f
    else:
        STORAGE = (s + 2*size + (approx.max_order + 2)*L)*f
    STORAGE = STORAGE / 2**30
    print("Space Complexity: ", STORAGE, " GB")
    print("(Store - Calculate) = ", s*f*(1 + 2*(L/s) - d/f))

    if "verbose" in opt:
        print("L, Map size, L/Map Size: ", L, s, L/s)
        print()
        print(ispc_code)
    #####################################################
    #print(ispc_code)
    GFLOPS, t = run(approx, ispc_code, size, num_samples, True)


    print()

    return STORAGE, GFLOPS


def test(a, b, orders, precisions):
    # Function used to obtain results. DONT CHANGE
    size, num_samples = 2**26, 50
    opt = ["arrays", "map", "random"]
    for precision in precisions:
        for order in orders:
            print(order, precision)
            if precision > 1e-7:
                name = "./approximations/32o" + str(order) + "-p" + str(precision)
                approx = adapt_i.load_from_file(name)
                print(name)
                run_one(approx, size, num_samples, opt)
                #run_one(approx, size, num_samples, opt + ["scalar"])
                run_one(approx, size, num_samples, opt + ["calc intervals"])
                #run_one(approx, size, num_samples, opt + ["scalar", "calc intervals"])


            name = "./approximations/64o" + str(order) + "-p" + str(precision)
            approx = adapt_i.load_from_file(name)
            print(name)
            run_one(approx, size, num_samples, opt)
            #run_one(approx, size, num_samples, opt + ["scalar"])
            run_one(approx, size, num_samples, opt + ["calc intervals"])
            #run_one(approx, size, num_samples, opt + ["scalar", "calc intervals"])



def save_approximations(a, b, orders, precisions):
    # Change dtypes and precisions manually
    size, num_samples = 2**12, 2
    opt = ["arrays", "map", "random"]
    for precision in precisions:
        for order in orders:
            print(order, precision)
            if precision > 1e-7:
                try:
                    name = "./approximations/32o" + str(order) + "-p" + str(precision)
                    approx = adapt_i.make_interpolant(a, b, f, order, 
                                                      precision, 'chebyshev', 
                                                      dtype=32, optimizations=opt)
                    adapt_i.write_to_file(name, approx)
                    run_one(approx, size, num_samples, opt)
                    run_one(approx, size, num_samples, opt + ["scalar"])
                    run_one(approx, size, num_samples, opt + ["calc intervals"])
                    run_one(approx, size, num_samples, opt + ["scalar", "calc intervals"])
                except:
                    pass

            opt = ["arrays", "map", "calc intervals", "random"]
            name = "./approximations/64o" + str(order) + "-p" + str(precision)
            approx = adapt_i.make_interpolant(a, b, f, order, 
                                              precision, 'chebyshev', 
                                              dtype=64, optimizations=opt)
            adapt_i.write_to_file(name, approx)
            run_one(approx, size, num_samples, opt)
            run_one(approx, size, num_samples, opt + ["scalar"])
            run_one(approx, size, num_samples, opt + ["calc intervals"])
            run_one(approx, size, num_samples, opt + ["scalar", "calc intervals"])



def test_remez_incorrect():
    # tests the lookup table size for incorrect remez algorithm and polynomial interpolation
    a, b = 0, 20
    order, precision = 6, 1e-6
    opt = ["arrays", "map", "calc intervals", "random", "remez incorrect"]
    approx = adapt_i.make_interpolant(a, b, f, order, 
                                      precision, 'chebyshev', 
                                      dtype=32, optimizations=opt)
    #adapt_i.write_to_file("./testingclass", approx)
    #approx = adapt_i.load_from_file("./testingclass")
    run_one(approx, opt=opt)


    opt = ["arrays", "map", "calc intervals", "random"]
    approx1 = adapt_i.make_interpolant(a, b, f, order, 
                                      precision, 'chebyshev', 
                                      dtype=32, optimizations=opt)
    run_one(approx1, opt=opt)


    opt = ["arrays", "map", "calc intervals", "random"]
    approx2 = adapt_i.make_interpolant(a, b, f, order, 
                                      precision, 'chebyshev', 
                                      dtype=32, optimizations=opt,
                                      adapt_type="Trivial")
    run_one(approx2, opt=opt)


    print("Incorrect Remez, Correct, Polynomial Interpolation")
    print(len(approx.map), len(approx1.map), len(approx2.map))
    print('{0:.16f}'.format(la.norm(approx.coeff,2)), 
          '{0:.16f}'.format(la.norm(approx1.coeff,2)), 
          '{0:.16f}'.format(la.norm(approx2.coeff,2)))
    print('{0:.16f}'.format(la.norm(approx.coeff,np.inf)), 
          '{0:.16f}'.format(la.norm(approx1.coeff,np.inf)), 
          '{0:.16f}'.format(la.norm(approx2.coeff,np.inf)))



def scalar_test():
    # decreasing the size causes the GFLOPS to go down...
    # size of 0 takes about 1e-5 seconds to run function.
    # with 2**10 and 2**15 size its still about that. 
    # 2**20 is better but 2**26 guarentees its good
    # takes long enough for the measurement to make sense.
    a, b = 1, 21
    order, precision = 3, 1e-3#np.finfo(np.float32).eps*10
    size, num_samples = 2**25, 50
    d = 32
    opt = ["arrays", "map", "random"]
    approx = adapt_i.make_interpolant(a, b, f, order, 
                                      precision, 
                                      'chebyshev', 
                                      dtype=d, 
                                      optimizations=opt)

    run_one(approx, size, num_samples, opt=opt + ["calc intervals"])
    run_one(approx, size, num_samples, opt=opt)
    # scalar does something incorrect? oh.. data race?
    run_one(approx, size, num_samples, opt=opt + ["scalar", "calc intervals"])
    run_one(approx, size, num_samples, opt=opt + ["scalar"])


# run the main program
if __name__ == "__main__":
    #scalar_test()
    #get_asm()
    #new_test()
    #test_remez_incorrect()
    # Function used to obtain results. DONT CHANGE
    # FAILS in case of Double precision near machine precision.
    # but only with calc intervals. Something is wrong with that.
    # not sure what it is though.
    # really fails by zeros. 1.72, but has too high of error on whole interval

    # x_scaled is correct. so maybe its something about the coefficients?
    # maybe im using the wrong dtype somewhere?
    # its actually not. The scaling/L is imprecise for some reason..
    #2/(b-a) is accurate though.. at least I figured it out...
    if 1:
        order, num_samples = 3, 5
        size = 2**23
        precision = 100*np.finfo(np.float64).eps
        opt = ["arrays", "map", "graph"]
        name = "./approximations/64o" + str(order) + "-p" + str(precision)
        approx = adapt_i.load_from_file(name)
        print(2*approx.D + approx.lgD)
        scaling = (approx.upper_bound -  approx.lower_bound) / len(approx.map)
        c = list(map(lambda x: (int(     bin(x)[           :-2*approx.D], 2),
                                int("0b"+bin(x)[-2*approx.D:  -approx.D], 2),
                                int("0b"+bin(x)[  -approx.D:           ], 2), 
                                         bin(x),
                                int("0b"+bin(x)[-2*approx.D:  -approx.D], 2)*scaling + approx.lower_bound,
                                int(     bin(x)[           :-2*approx.D], 2)*scaling,
                            ), approx.cmap))
        """
        print(approx.lower_bound, approx.upper_bound)
        print("   L"," l", "leaf index")
        for a in c[:20]:
            print(a[0], a[1]*scaling, "\t",a[2],"\t",a[4], "\t", a[5])
        print(2*approx.D + approx.lgD)
        print(approx.cmap.dtype)
        print(len(approx.map))
        print((approx.upper_bound - approx.lower_bound)/len(approx.cmap))
        print(1/((approx.upper_bound - approx.lower_bound)/len(approx.cmap)))
        print(2./((approx.upper_bound - approx.lower_bound)/len(approx.cmap)))
        """
        #print(approx.interval_a)
        #print(approx.interval_b)
        print(precision)
        run_one(approx, size, num_samples, opt)
        run_one(approx, size, num_samples, opt + ["scalar"])
        run_one(approx, size, num_samples, opt + ["calc intervals"])
        run_one(approx, size, num_samples, opt + ["scalar", "calc intervals"])

    if 0:
        a, b = 1, 21
        orders = [5,3]
        precisions = [100*np.finfo(np.float32).eps, 100*np.finfo(np.float64).eps]
        #save_approximations(a, b, orders, precisions)
        test(a, b, orders, precisions)


