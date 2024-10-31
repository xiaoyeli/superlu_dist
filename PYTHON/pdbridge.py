import os
import ctypes
import sys
from sys import platform

def setup_pdbridge(sp, INT64):
    # Define the function signatures as shown in your original code
    sp.pdbridge_init.restype = None
    if INT64 == 0:
        sp.pdbridge_init.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p)
        ]
    else:
        sp.pdbridge_init.argtypes = [
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
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
    if DLLFOUND:
        sp = ctypes.cdll.LoadLibrary(DLL)
        setup_pdbridge(sp, INT64)
        return sp
    else:
        raise Exception("Cannot find the superlu_dist_python library. Try to set the SUPERLU_PYTHON_LIB_PATH environment variable correctly.")