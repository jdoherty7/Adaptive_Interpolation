import os
import numpy as np
import numpy.linalg as la
import ctypes
import ctypes.util
from time import time
from tempfile import TemporaryDirectory



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



def make_code(experiment):
    if experiment == "triad":
        ispc_code = """
        export void stream(
                    uniform double *uniform a, 
                    uniform double *uniform b, 
                    uniform double *uniform c, 
                    uniform double scalar, 
                    uniform int32 n){
            for (varying int i=programIndex; i<n; i+=programCount){
                a[i] = b[i] + scalar * c[i];
            }
        }
        """
    elif experiment == "copy":
        ispc_code = """
        export void stream(
                    uniform double *uniform a, 
                    uniform double *uniform b, 
                    uniform int32 n){
            for (varying int i=programIndex; i<n; i+=programCount){
                a[i] = b[i];
            }
        }
        """
    elif experiment == "scale":
        ispc_code = """
        export void stream(
                    uniform double *uniform a, 
                    uniform double *uniform b, 
                    uniform double scalar, 
                    uniform int32 n){
            for (varying int i=programIndex; i<n; i+=programCount){
                a[i] = scalar * b[i];
            }
        }
        """
    elif experiment == "sum":
        ispc_code = """
        export void stream(
                    uniform double *uniform a, 
                    uniform double *uniform b,
                    uniform double *uniform c,  
                    uniform int32 n){
            for (varying int i=programIndex; i<n; i+=programCount){
                a[i] = b[i] + c[i];
            }
        }
        """
    return ispc_code


NRUNS = 20
ARRAY_SIZE = 2**26


STREAM_DTYPE = np.float64
STREAM_CTYPE = ctypes.c_float
INDEX_DTYPE = np.int32
INDEX_CTYPE = ctypes.c_int

# core pinning, frequency scaling.
# cache line is replaced
# read the cacheline then you can write it to memory
# streaming_store is when
# ispc streaming store patch which allows it to do it.
# issue port - sandy bridge architecture article
def main(experiment):
    print()
    print("Task: ", experiment)
    with open("tests/tasksys.cpp", "r") as ts_file:
        tasksys_source = ts_file.read()

    ispc_code = make_code(experiment)

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
                    #"--opt=force-aligned-memory",
                    "--opt=disable-loop-unroll",
                    #"--opt=fast-math",
                    #"--woff",
                    #"--opt=disable-fma",
                    "--addressing=32",
                    ]
                    ),
                ispc_bin= "/home/jjdoher2/Desktop/ispc-v1.9.1-linux/ispc",
                quiet=True,
                )

        knl_lib = ctypes.cdll.LoadLibrary(os.path.join(tmpdir, "shared.so"))


        scalar = 4
        a =  2*np.ones(ARRAY_SIZE, dtype=STREAM_DTYPE)
        b =  3*np.ones(ARRAY_SIZE, dtype=STREAM_DTYPE)
        c = -5*np.ones(ARRAY_SIZE, dtype=STREAM_DTYPE)
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

        for i in range(2):
            call_kernel()


        ts = []
        for irun in range(NRUNS):
            start_time = time()
            for i in range(10):
                call_kernel()
            elapsed = time() - start_time
            ts.append(elapsed/10)
        ts = np.array(ts)
        print(ts)
        print("Min Time: ", np.min(ts))
        print("Max Time: ", np.max(ts))
        print("Avg Time: ", np.mean(ts))
        by = 3 if experiment in ["triad", "sum"] else 2
        print("Max MB: ", 1e-9*by*a.nbytes/np.min(ts), "GB/s")
        print("Min MB: ", 1e-9*by*a.nbytes/np.max(ts), "GB/s")
        print("Avg MB: ", 1e-9*by*a.nbytes/np.mean(ts), "GB/s")

        #assert la.norm(a-b+scalar*c, np.inf) < np.finfo(STREAM_DTYPE).eps * 10


if __name__ == "__main__":
    main("triad")
    main("copy")
    main("scale")
    main("sum")

