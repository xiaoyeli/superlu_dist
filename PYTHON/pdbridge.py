import os
import ctypes
import sys
from sys import platform
import time
import pickle
import numpy as np

def setup_pdbridge(sp, INT64):
    # Define the function signatures as shown in your original code
    sp.sizeof_int_t.argtypes = None
    sp.sizeof_int_t.restype = ctypes.c_int
    int_t = sp.sizeof_int_t()
    
    sp.pdbridge_init.restype = None
    if INT64 == 0:
        if int_t != 4:
            raise Exception("libsuperlu_dist_python has been compiled with 64-bit integers. Please recompile it with 32-bit integers or use INT64=1 in the python APIs.")

        sp.pdbridge_init.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p)
        ]
    else:
        if int_t != 8:
            raise Exception("libsuperlu_dist_python has been compiled with 32-bit integers. Please recompile it with 64-bit integers or use INT64=0 in the python APIs.")

        sp.pdbridge_init.argtypes = [
            ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p)
        ]

    sp.pdbridge_factor.restype = None
    sp.pdbridge_factor.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    sp.pdbridge_solve.restype = None
    sp.pdbridge_solve.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    sp.pdbridge_logdet.restype = None
    sp.pdbridge_logdet.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)]
    sp.pdbridge_free.restype = None
    sp.pdbridge_free.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

def load_library(INT64):
    # Check platform and set library extension
    if platform == "linux" or platform == "linux2":
        pos = '.so'
    elif platform == "darwin":
        pos = '.dylib'
    elif platform == "win32":
        raise Exception("Windows is not yet supported")

    DLLFOUND = False
    INSTALLDIR = os.getenv('SUPERLU_PYTHON_LIB_PATH')

    DLL = os.path.abspath(__file__ + "/../../") + '/libsuperlu_dist_python' + pos
    if os.path.exists(DLL):
        DLLFOUND = True
    elif INSTALLDIR is not None:
        DLL = os.path.join(INSTALLDIR, 'libsuperlu_dist_python' + pos)
        if os.path.exists(DLL):
            DLLFOUND = True
    else:
        DLL = os.path.join('./libsuperlu_dist_python' + pos)
        if os.path.exists(DLL):
            DLLFOUND = True            
    if DLLFOUND:
        sp = ctypes.cdll.LoadLibrary(DLL)
        setup_pdbridge(sp, INT64)
        return sp
    else:
        raise Exception("Cannot find the superlu_dist_python library. Try to set the SUPERLU_PYTHON_LIB_PATH environment variable correctly.")




###################################################################################################
###########  define the APIs

def wait_for_flag(expected_flag, control_file, poll_interval=0.001):
    """Poll the control file until its content equals the expected flag."""
    while True:
        if os.path.exists(control_file):
            with open(control_file, "r") as f:
                flag = f.read().strip()
            if flag == expected_flag:
                return True
        time.sleep(poll_interval)
        

####################### initialization and factorization
def superlu_factor(KV, INT64=1, algo3d=0, verbosity=False):
    start = time.time()
    CONTROL_FILE=os.getenv("CONTROL_FILE", "control.txt")

    # The following if test makes sure superlu cleans up the factorization is there is one 
    if os.path.exists(CONTROL_FILE):
        with open(CONTROL_FILE, "r") as f:
            flag = f.read().strip()
        if flag != "clean":
            superlu_freeLU(verbosity)

    DATA_FILE=os.getenv("DATA_FILE", "data.bin")
    with open(DATA_FILE, "wb") as f:
        pickle.dump((KV,INT64,algo3d), f)
    with open(CONTROL_FILE, "w") as f:
        f.write("init")
    wait_for_flag("done", CONTROL_FILE)
    end = time.time()
    if verbosity==True:
        print(f"Time spent in pdbridge_init: {end - start} seconds")
    
    start = time.time()
    with open(CONTROL_FILE, "w") as f:
        f.write("factor")
    wait_for_flag("done", CONTROL_FILE)
    end = time.time()
    if verbosity==True:
        print(f"Time spent in pdbridge_factor: {end - start} seconds")


####################### solve 
def superlu_solve(vec, verbosity=False):
    vec = np.asarray(vec)  
    if vec.ndim == 1:
        vec = vec.reshape(-1, 1)  
    nrhs=vec.shape[-1]
    start = time.time()
    CONTROL_FILE=os.getenv("CONTROL_FILE", "control.txt")
    DATA_FILE=os.getenv("DATA_FILE", "data.bin")
    RESULT_FILE=os.getenv("RESULT_FILE", "result.bin")    
    with open(DATA_FILE, "wb") as f:
        pickle.dump((vec,nrhs), f)
    with open(CONTROL_FILE, "w") as f:
        f.write("solve")
    wait_for_flag("done", CONTROL_FILE)
    with open(RESULT_FILE, "rb") as f:
        vec_out = pickle.load(f)
        np.copyto(vec, vec_out)
    end = time.time()
    if verbosity==True:
        print(f"Time spent in pdbridge_solve: {end - start} seconds")


####################### log-determinant 
def superlu_logdet(verbosity=False):
    start = time.time()
    CONTROL_FILE=os.getenv("CONTROL_FILE", "control.txt")
    RESULT_FILE=os.getenv("RESULT_FILE", "result.bin")    
    with open(CONTROL_FILE, "w") as f:
        f.write("logdet")
    wait_for_flag("done", CONTROL_FILE)
    with open(RESULT_FILE, "rb") as f:
        sign,log_det = pickle.load(f)
    end = time.time()
    if verbosity==True:
        print(f"Time spent in pdbridge_logdet: {end - start} seconds")
    return sign,log_det

####################### free stuff
def superlu_freeLU(verbosity=False):
    start = time.time()
    CONTROL_FILE=os.getenv("CONTROL_FILE", "control.txt")
    with open(CONTROL_FILE, "w") as f:
        f.write("free")
    wait_for_flag("clean", CONTROL_FILE)
    end = time.time()
    if verbosity==True:
        print(f"Time spent in pdbridge_free: {end - start} seconds")


####################### terminate all workers if no more superLU calls are needed
def superlu_terminate(verbosity=False):
    start = time.time()
    CONTROL_FILE=os.getenv("CONTROL_FILE", "control.txt")
    with open(CONTROL_FILE, "w") as f:
        f.write("terminate")
    end = time.time()
    if verbosity==True:
        print(f"Time spent in pdbridge_terminate: {end - start} seconds")


