import os
import numpy as np
import numpy.linalg as la
import ctypes
import ctypes.util
from time import time
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.use("Agg")
#STREAM is intended to measure the bandwidth from main memory.  
#It can, of course, be used to measure cache bandwidth as well, but that is not what I have been 
#publishing at the web site.  Maybe someday.... 
#The general rule for STREAM is that each array must be at least 4x 
#the size of the sum of all the last-level caches used in the run, or 1 Million elements -- whichever is larger.


def address_from_numpy(obj):
    ary_intf = getattr(obj, "__array_interface__", None)
    if ary_intf is None:
        raise RuntimeError("no array interface")

    buf_base, is_read_only = ary_intf["data"]
    return buf_base + ary_intf.get("offset", 0)



def cptr_from_numpy(obj):
    return ctypes.c_void_p(address_from_numpy(obj))

# https://github.com/hgomersall/pyFFTW/blob/master/pyfftw/utils.pxi#L172
def align(array, dtype, order='C', n=64):
    '''empty_aligned(shape, dtype='float64', order='C', n=None)
    Function that returns an empty numpy array that is n-byte aligned,
    where ``n`` is determined by inspecting the CPU if it is not
    provided.
    The alignment is given by the final optional argument, ``n``. If
    ``n`` is not provided then this function will inspect the CPU to
    determine alignment. The rest of the arguments are as per
    :func:`numpy.empty`.
    '''
    shape = array.shape
    itemsize = np.dtype(dtype).itemsize

    # Apparently there is an issue with numpy.prod wrapping around on 32-bits
    # on Windows 64-bit. This shouldn't happen, but the following code
    # alleviates the problem.
    if not isinstance(shape, (int, np.integer)):
        array_length = 1
        for each_dimension in shape:
            array_length *= each_dimension

    else:
        array_length = shape

    base_ary = np.empty(array_length*itemsize+n, dtype=np.int8)

    # We now need to know how to offset base_ary
    # so it is correctly aligned
    _array_aligned_offset = (n-address_from_numpy(base_ary)) % n

    new_array = np.frombuffer(
                base_ary[_array_aligned_offset:_array_aligned_offset-n].data,
                dtype=dtype).reshape(shape, order=order)

    np.copyto(new_array, array)
    return new_array



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



def make_code(experiment, runs, single):
    if experiment == "triad":
        ispc_code = """
        export void stream(
                    uniform double *uniform a, 
                    uniform double *uniform b, 
                    uniform double *uniform c, 
                    uniform double scalar, 
                    uniform int32 n){
            for (uniform int32 runs=0; runs<%i; runs+=1){
                for (uniform int32 i=0; i<n; i+=programCount){
                    varying int32 is = i + programIndex;
                    // broadcast sends the value that i has for the program instance
                    // specified in the second argument to all other program instances
                    streaming_store(a + i, broadcast(b[i] + scalar * c[i], 0));
                    //a[is] = b[is] + scalar * c[is];
                }
            }
        }
        """ % runs
    elif experiment == "copy":
        ispc_code = """
        export void stream(
                    uniform double *uniform a, 
                    uniform double *uniform b, 
                    uniform int32 n){
            for (uniform int32 runs=0; runs<%i; runs+=1){
                for (uniform int32 i=0; i<n; i+=programCount){
                    varying int32 is = i + programIndex;
                    streaming_store(a+i, broadcast(b[i], 0));
                    //a[is] = b[is];
                }
            }
        }
        """% runs
    elif experiment == "scale":
        ispc_code = """
        export void stream(
                    uniform double *uniform a, 
                    uniform double *uniform b, 
                    uniform double scalar, 
                    uniform int32 n){
            for (uniform int32 runs=0; runs<%i; runs+=1){
                for (uniform int32 i=0; i<n; i+=programCount){
                    varying int32 is = i + programIndex;
                    streaming_store(a+i, broadcast(scalar * b[i], 0));
                    //a[is] = scalar * b[is];
                }
            }
        }
        """% runs
    elif experiment == "sum":
        ispc_code = """
        export void stream(
                    uniform double *uniform a, 
                    uniform double *uniform b,
                    uniform double *uniform c,  
                    uniform int32 n){
            for (uniform int32 runs=0; runs<%i; runs+=1){
                for (uniform int32 i=0; i<n; i+=programCount){
                    varying int32 is = i + programIndex;
                    streaming_store(a+i, broadcast(b[i] + c[i], 0));
                    //a[is] = b[is] + c[is];
                }
            }
        }
        """% runs
    if single==True:
        ispc_code = ispc_code.replace("double", "float")
    return ispc_code




# core pinning, frequency scaling.
# cache line is replaced
# read the cacheline then you can write it to memory
# streaming_store is when
# ispc streaming store patch which allows it to do it.
# issue port - sandy bridge architecture article
def main(experiment):

    ALIGN_TO = 32
    # 22 is the first above the L3, its double the L3 about 50,000 KB
    sizes = np.power(2, np.arange(5, 26))
    single=True


    #ARRAY_SIZE = [size(L1)/3, 3*size(L3)]
    """
    L1d cache:            32K – data cache
    L1i cache:            32K – instruction cache
    L2 cache:             256K
    L3 cache:             30720K
    cache size:           30720 KB
    """

    if single:
        STREAM_DTYPE = np.float32
        STREAM_CTYPE = ctypes.c_float
        INDEX_DTYPE = np.int32
        INDEX_CTYPE = ctypes.c_int
    else:
        STREAM_DTYPE = np.float64
        STREAM_CTYPE = ctypes.c_double
        INDEX_DTYPE = np.int32
        INDEX_CTYPE = ctypes.c_int

    KBs = []
    Bandwidth = []
    for ARRAY_SIZE in sizes:
        #NRUNS * ARRAY_SIZE = 10* 2**26
        NRUNS = int((50 * 2**26)/ARRAY_SIZE)
        print()
        print("Task: ", experiment)
        with open("tests/tasksys.cpp", "r") as ts_file:
            tasksys_source = ts_file.read()

        ispc_code = make_code(experiment, NRUNS, single)

        with TemporaryDirectory() as tmpdir:
            #print(ispc_code)

            build_ispc_shared_lib(
                    tmpdir,
                    [("stream.ispc", ispc_code)],
                    [("tasksys.cpp", tasksys_source)],
                    cxx_options=["-g", "-fopenmp", "-DISPC_USE_OMP"],
                    ispc_options=([
                        "-g", "-O1", "--no-omit-frame-pointer",
                        "--target=avx2-i32x16",
                        "--opt=force-aligned-memory",
                        "--opt=disable-loop-unroll",
                        #"--opt=fast-math",
                        #"--woff",
                        #"--opt=disable-fma",
                        "--addressing=32",
                        ]
                        ),
                    #ispc_bin= "/home/ubuntu-boot/Desktop/ispc-v1.9.1-linux/ispc",
                    ispc_bin= "/home/ubuntu-boot/Desktop/ispc-1.9-with-streaming-store/ispc",
                    quiet=True,
                    )

            knl_lib = ctypes.cdll.LoadLibrary(os.path.join(tmpdir, "shared.so"))


            scalar = 4
            choice ={   "triad":(1, 3, 0, 7), 
                        "copy": (1, 9,-1,-1), 
                        "scale":(), 
                        "sum":  ()
                    }

            a0, b0, c0, scalar = choice[experiment]
            a = a0*np.ones(ARRAY_SIZE, dtype=STREAM_DTYPE)
            b = b0*np.ones(ARRAY_SIZE, dtype=STREAM_DTYPE)
            c = c0*np.ones(ARRAY_SIZE, dtype=STREAM_DTYPE)

            a = align(a, dtype=STREAM_DTYPE)#, n=ALIGN_TO)
            b = align(b, dtype=STREAM_DTYPE)#, n=ALIGN_TO)
            c = align(c, dtype=STREAM_DTYPE)#, n=ALIGN_TO)
            g = knl_lib.stream

            if experiment == "copy":
                x = [cptr_from_numpy(a), 
                    cptr_from_numpy(b),
                    INDEX_CTYPE(ARRAY_SIZE),]
            elif experiment == "triad":
                x = [cptr_from_numpy(a), 
                    cptr_from_numpy(b), 
                    cptr_from_numpy(c),
                    STREAM_CTYPE(scalar),
                    INDEX_CTYPE(ARRAY_SIZE),]
            elif experiment == "scale":
                x = [cptr_from_numpy(a), 
                    cptr_from_numpy(b), 
                    STREAM_CTYPE(scalar),
                    INDEX_CTYPE(ARRAY_SIZE),]
            elif experiment == "sum":
                x = [cptr_from_numpy(a), 
                    cptr_from_numpy(b), 
                    cptr_from_numpy(c),
                    INDEX_CTYPE(ARRAY_SIZE),]

            for i in range(2):
                g(*x)

            def call_kernel():
                g(*x)

            for i in range(4):
                call_kernel()


            ts = []
            start_time = time()
            # This will run Nruns # of times
            call_kernel()
            elapsed = time() - start_time
            ts.append(elapsed/NRUNS)
            ts = np.array(ts)
            #print(ts)
            #print("Min Time: ", np.min(ts))
            #print("Max Time: ", np.max(ts))
            #print("Avg Time: ", np.mean(ts))
            by = 3 if experiment in ["triad", "sum"] else 2
            # The STREAM BENCHMARK paper considers KB=1024 and GB=2^30
            GB = 1e-9*by*a.nbytes
            KB = 1e-3*by*a.nbytes
            print("KB: ", KB)
            KBs.append(KB)
            # only care about maximum bandwidth
            Bandwidth.append(GB/np.min(ts))


            print("Max MB: ", GB/np.min(ts), "GB/s")
            #print("Min MB: ", GB/np.max(ts), "GB/s")
            #print("Avg MB: ", GB/np.mean(ts), "GB/s")

            #print("Max Error")
            if experiment == "triad":
                error = la.norm(a-b-scalar*c, np.inf)
            elif experiment == "copy":
                error = la.norm(a-b         , np.inf)
            elif experiment == "scale":
                error = la.norm(a-(b*scalar), np.inf)
            else:
                error = la.norm(a-b-c       , np.inf)
            assert error < 1e-1
        
    print()
    print("Single=",single)
    print(KBs)
    print("Bandwidths")
    print(Bandwidth)
    plt.figure()
    plt.title("Memory Bandwidth for '"+experiment+"' Test")
    
    plt.axvline(x=32,    color="r", label="End of L1")
    plt.axvline(x=256,   color="b", label="End of L2")
    plt.axvline(x=30720, color="g", label="End of L3")

    plt.plot(KBs, Bandwidth, c="k", label=experiment)

    plt.xscale("log")
    plt.xlabel("Memory Used (KB)")
    plt.ylabel("Memory Bandwidth (GB/s)")
    plt.legend()
    plt.savefig(experiment+str(single)".png")

if __name__ == "__main__":
    main("triad")
    main("copy")
    main("scale")
    main("sum")

