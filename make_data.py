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
#mpl.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import adaptive_interpolation.adapt as adapt
import adaptive_interpolation.approximator as app
import adaptive_interpolation.generate as generate
import adaptive_interpolation.adaptive_interpolation as adapt_i
import loopy as lp
from loopy.tools import (empty_aligned, address_from_numpy,
        build_ispc_shared_lib, cptr_from_numpy)

def transform(knl, vars, stream_dtype):
    vars = [v.strip() for v in vars.split(",")]
    knl = lp.assume(knl, "n>0")
    knl = lp.split_iname(
        knl, "i", 2**18, outer_tag="g.0", slabs=(0, 1))
    knl = lp.split_iname(knl, "i_inner", 8, inner_tag="l.0")

    knl = lp.add_and_infer_dtypes(knl, {
        var: stream_dtype
        for var in vars
        })

    knl = lp.set_argument_order(knl, vars + ["n"])

    return knl

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

def save(name, array, ntype):
    new_file = open(name, "w")
    for element in array:
        new_file.write(str(ntype(element))+",")
    new_file.close()


def make_data(size, order, precision, d, vectorized=True, approx=None, code=None, opt=[]):
    a, b = 0, 20
    if code==None and approx==None:
        approx = adapt_i.make_interpolant(a, b, f, order,
                                          precision, 'chebyshev', dtype=d, optimizations=opt)

        # see how much time to process array, vector width = 1
        if vectorized:
            code = adapt_i.generate_code(approx, size=size, vector_width=8, cpu=True)
        else:
            code = adapt_i.generate_code(approx, size=size, cpu=True)
    #print(code)
    dt = approx.dtype_name
    var = "uniform " if vectorized else ""
    header = "export void eval("
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
        header += "const " + var + dt + " f[], "

   
    header += var + dt + " x[], " + var + dt + " y[])"


    code = code.replace("\n", "\n\t")
    full_code = header + "{\n\n" + code + "\n}"
    with open("simple_ispc.ispc", "w") as h:
        h.writelines(full_code)

    s = [size, size]
    if "arrays" in opt:
        s.append(len(approx.mid))
        s.append(len(approx.left))
        s.append(len(approx.right))
        s.append(len(approx.interval_a))
        s.append(len(approx.interval_b))
        s.append(len(approx.coeff))
    else:
        s.append(len(approx.tree_1d))

    if "map" in opt:
        s.append(len(approx.map))

    dt = float
    if "arrays" in opt:
        save("left.txt", approx.left, dt)
        save("right.txt", approx.right, dt)
        save("interval_a.txt", approx.interval_a, dt)
        save("interval_b.txt", approx.interval_b, dt)
        save("coeff.txt", approx.coeff, dt)
        save("mid.txt", approx.mid, dt)
        os.system("mv left.txt tests/perf/")
        os.system("mv right.txt tests/perf/")
        os.system("mv interval_a.txt tests/perf/")
        os.system("mv interval_b.txt tests/perf/")
        os.system("mv coeff.txt tests/perf/")
        os.system("mv mid.txt tests/perf/")
    else:     
        save("tree.txt", approx.tree_1d, dt)
        os.system("mv tree.txt tests/perf/")

    if "map" in opt:
        save("f.txt", approx.map, dt)
        os.system("mv f.txt tests/perf/")

    # add changing eval function in tests/perf/simple.cpp
    """
    with open("tests/perf/simple.cpp", 'rw') as sim:
        cpp = sim.readlines()
        # actually this only needs to comment and uncomment correct lines 
        if "arrays" in opt or "map" in opt:
            cpp.replace("", "eval(mid, left, right, interval_a, interval_b, coeff, f, x, y);\n")
        else:
            cpp.repalce("", "eval(tree, x, y);")
        sim.writelines(cpp)
    """

    save("sizes.txt", s, int)
    os.system("mv sizes.txt tests/perf/")
    os.system("mv *.ispc tests/perf/")
    return approx, full_code


def run_data(tree_depth, order, size, n, vec=True):
    if vec:
        flop = size*(4 + 2 + 2*(order-2))
    else:
        #flop = size*(5 + 3 + 5*(order-2))
        # with fused mult add / sub and if 2*x_scaled is done outside loop
        flop = size*(4 + 2 + 2*(order-2))
    memop = size*(4*tree_depth + order + 4)*4 # 4 bytes each access (single precision)

    #os.system("cd tests/perf && make > junk.txt")
    dts = []
    for i in range(n):
        start = time.time()
        # this takes about .004 seconds to start up, so theres about 3 digits of accuracy if dt~1
        #os.system("cd tests/perf && ./simple > junk.txt")
        dts.append(time.time() - start)
    dt_std = np.std(dts)
    dt_mean = np.mean(dts)
    GFLOPS_mean = flop/(dt_mean*2**30)
    GFLOPS_std = flop/((dt_mean - dt_std)*2**30) - GFLOPS_mean
    GFLOPS_std = (GFLOPS_std + (GFLOPS_mean - flop/((dt_mean + dt_std)*2**30)))/2

    mem_bandwidth_mean = n*memop/(dt_mean*2**30)
    mem_bandwidth_std = n*memop/((dt_mean - dt_std)*2**30) - mem_bandwidth_mean
    print("dt mean, dt std: ",dt_mean, dt_std)
    return GFLOPS_mean, GFLOPS_std, mem_bandwidth_mean, mem_bandwidth_std


def run(approx, code, size, NRUNS):
    ALIGN_TO = 4096
    if approx.dtype_name == "float":
        dt = np.float32
        STREAM_DTYPE = np.float32
        STREAM_CTYPE = ctypes.c_float
        INDEX_DTYPE = np.int32
        INDEX_CTYPE = ctypes.c_int
    elif approx.dtype_name == "double":
        dt = np.float64
        STREAM_DTYPE = np.float64
        STREAM_CTYPE = ctypes.c_doublev
        INDEX_DTYPE = np.int64
        INDEX_CTYPE = ctypes.c_longlong


    with open("tests/tasksys.cpp", "r") as ts_file:
        tasksys_source = ts_file.read()  

    with TemporaryDirectory() as tmpdir:
        print(code)
        build_ispc_shared_lib(
                tmpdir,
                [("stream.ispc", code)],
                [("tasksys.cpp", tasksys_source)],
                cxx_options=["-g", "-fopenmp", "-DISPC_USE_OMP"],
                ispc_options=([
                    "-g", "--no-omit-frame-pointer",
                    #"--target=avx2-i32x8",
                    "--arch=x86-64",
                    "--target=avx2",
                    #"--opt=force-aligned-memory",
                    "--opt=disable-loop-unroll",
                    #"--opt=fast-math",
                    "--opt=disable-fma",
                    "--woff"
                    ]
                    + (["--addressing=64"] if INDEX_DTYPE == np.int64 else [])
                    ),
                ispc_bin="/home/ubuntu-boot/Desktop/ispc-v1.9.1-linux/ispc",
                quiet=True,
                )



        x = np.linspace(approx.lower_bound, approx.upper_bound, size, dtype=dt)
        y = np.zeros(size, dtype=dt)
        approx.tree_1d = np.array(approx.tree_1d, dtype=dt)
        approx.map = np.array(approx.map, dtype=dt)
        approx.mid = np.array(approx.mid, dtype=dt)
        approx.left = np.array(approx.left, dtype=dt)
        approx.right = np.array(approx.right, dtype=dt)
        approx.interval_a = np.array(approx.interval_a, dtype=dt)
        approx.interval_b = np.array(approx.interval_b, dtype=dt)
        approx.coeff  = np.array(approx.coeff, dtype=dt)


        knl_lib = ctypes.cdll.LoadLibrary(os.path.join(tmpdir, "shared.so"))


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
        for i in range(10):
            call_kernel()
        #print(y)
        #print(np.max(y))

        plt.figure()
        plt.plot(x[::2048],y[::2048])
        plt.show()

        start_time = time.time()
        for irun in range(NRUNS):
            call_kernel()
        elapsed = time.time() - start_time

        FLOPS = (4 + 2 + 2*(approx.max_order-2))
        print("Average Runtime:", elapsed/NRUNS)
        # times size*4 because thats the number of bytes in x
        GFLOPS = FLOPS*size/(2**30)
        print(GFLOPS/elapsed, "GFLOPS/s")
        return y


def main(opt=[]):
    # 2**30 is a GigaByte
    # 2**40 is a TerraByte
    # uh oh, it seems like half, 2.84 seconds is taken in the c++ script...
    # yup, i get a 2x speed up when building the library
    # guess i need to make a shared library
    print(opt)
    num_samples = 2
    size = 2**26#8*2**26
    with_scalar = False
    orders = [5]#np.linspace(2, 10, 3)
    precisions = [1e-6]
    for precision in precisions:
        gflops_scalar, gflops_vect = [], []
        gflops_scalar_err, gflops_vect_err = [], []
        mb_scalar, mb_vect = [], []
        mb_scalar_err, mb_vect_err = [], []
        for order in orders:
            print("Vector: ", order, precision)
            approx, ispc_code = make_data(size, order, precision, '32', vectorized=True, opt=opt)
            print("tree levels: ", approx.num_levels)
            #y = run(approx, ispc_code, size, num_samples)
            #g, gs, mb, mbs = run_data(approx.num_levels, order, size, num_samples)
            #mb_vect.append(mb)
            #mb_vect_err.append(mbs)
            #gflops_vect.append(g)
            #gflops_vect_err.append(gs)

            if with_scalar:
                print("Scalar: ", order)
                code = adapt_i.generate_code(approx, size, None, cpu=True)
                approx, ispc_code = make_data(size, order, precision, '32', vectorized=False, approx=approx, code=code)
                y = run(approx, ispc_code, size, num_samples)
                #g, gs, mb, mbs = run_data(approx.num_levels, order, size, num_samples, vec=False)
                #mb_scalar.append(mb)
                #mb_scalar_err.append(mbs)
                #gflops_scalar.append(g)
                #gflops_scalar_err.append(gs)
        """
        print()
        print("GFLOPS:", gflops_vect)
        if with_scalar:
            print(gflops_scalar)
        print(gflops_vect_err)
        print()
        #if with_scalar:
        #    print(gflops_scalar_err)
        #print()
        #print()
        print("Memory Bandwidth", mb_vect)
        print(mb_vect_err)
        #print(mb_scalar)
        """
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




# run the main program
if __name__ == "__main__":
    #main()
    main(opt=['test first loop'])
    #main(opt=['test second loop'])
    #main(opt=['none'])
    #main(opt=['trim_data'])
    #main(opt=['trim_data', 'unroll'])
    #main(opt=['trim_data', 'unroll', 'unroll_order'])
    #main(opt=['trim_data', 'unroll', 'unroll_order', 'prefetch'])
    #main(opt=['arrays', 'map'])

