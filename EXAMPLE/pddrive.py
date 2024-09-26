import numpy as np
import os
import ctypes
import scipy
from scipy.sparse.linalg import splu
import time
import sys
import mpi4py
from mpi4py import MPI



def setup_pdbridge(sp,INT64):
    # Define the function signature for pdbridge_init
    sp.pdbridge_init.restype = None
    if(INT64==0):
        sp.pdbridge_init.argtypes = [
            ctypes.c_int,                                   # int_t m
            ctypes.c_int,                                   # int_t n
            ctypes.c_int,                                   # int_t nnz
            ctypes.POINTER(ctypes.c_int),                   # int_t *rowind
            ctypes.POINTER(ctypes.c_int),                   # int_t *colptr
            ctypes.POINTER(ctypes.c_double),                # double *nzval
            ctypes.POINTER(ctypes.c_void_p),                # void **pyobj
            ctypes.c_int,                                   # int argc
            ctypes.POINTER(ctypes.c_char_p)                 # char *argv[]
        ]
    else:
        sp.pdbridge_init.argtypes = [
            ctypes.c_int64,                                   # int_t m
            ctypes.c_int64,                                   # int_t n
            ctypes.c_int64,                                   # int_t nnz
            ctypes.POINTER(ctypes.c_int64),                   # int_t *rowind
            ctypes.POINTER(ctypes.c_int64),                   # int_t *colptr
            ctypes.POINTER(ctypes.c_double),                # double *nzval
            ctypes.POINTER(ctypes.c_void_p),                # void **pyobj
            ctypes.c_int,                                   # int argc
            ctypes.POINTER(ctypes.c_char_p)                 # char *argv[]
        ]


    sp.pdbridge_factor.restype = None
    sp.pdbridge_factor.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),                # void **pyobj
    ]

    sp.pdbridge_solve.restype = None
    sp.pdbridge_solve.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),                # void **pyobj
        ctypes.c_int,                                   # int nrhs
        ctypes.POINTER(ctypes.c_double),                # double *b_global
    ]
    # Define the function signature for pdbridge_logdet
    sp.pdbridge_logdet.restype = None
    sp.pdbridge_logdet.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),                # void **pyobj
        ctypes.POINTER(ctypes.c_int),                   # int* sign
        ctypes.POINTER(ctypes.c_double),                # double *logdet
    ]
    # Define the function signature for pdbridge_free
    sp.pdbridge_free.restype = None
    sp.pdbridge_free.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),                # void **pyobj
    ]



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if(rank==0):
    print('mpi4py version: ', mpi4py.__version__)
    print('MPI count:', size)


####################### create the matrix
INT64 = 1 # whether to use 64bit integer (requring superlu_dist to be compiled with 64-bit indexing)
rng = np.random.default_rng()
n = 1000
nrhs = 1

if(rank==0):
    a = scipy.sparse.random(n, n, density=0.01, random_state=rng)
    m = (a.T @ a) + scipy.sparse.identity(n)
    print("sparsity: ", float(m.nnz)/n**2)
else: #dummy data, not refered inside superlu_dist API
    a = scipy.sparse.random(1, 1, density=1, random_state=rng)
    m = (a.T @ a) 
m_csc = m.tocsc()

if(INT64==0):
    rowind = m_csc.indices.astype(np.int32)
    colptr = m_csc.indptr.astype(np.int32) 
    nzval = m_csc.data.astype(np.float64)  
    nnz=m.nnz
else:
    rowind = m_csc.indices.astype(np.int64)
    colptr = m_csc.indptr.astype(np.int64) 
    nzval = m_csc.data.astype(np.float64)  
    nnz=np.int64(m.nnz)
    n = np.int64(n)

####################### handle options 
argv=sys.argv
if(len(argv)==1): # options are not passed via command line, set them manually here. If they are not set here, default values are used
    argv.extend(['-r', '%i'%(np.sqrt(size))])  # process rows
    argv.extend(['-c', '%i'%(np.sqrt(size))])  # process columns
    argv.extend(['-p', '1'])  # row permutation 
    argv.extend(['-q', '2'])  # column permutation 
    argv.extend(['-s', '0'])  # parallel symbolic factorization, needs -q to be 5
    argv.extend(['-i', '0'])  # whether to use iterative refinement 0, 1, 2
argc = len(argv)
if(rank==0):    
    print('SuperLU options: ',argv[1:])
argv_bytes = [arg.encode('utf-8') for arg in argv]
argv_ctypes = (ctypes.c_char_p * (argc + 1))(*argv_bytes, None)





####################### load DLL
import sys
from sys import platform
if platform == "linux" or platform == "linux2":
    pos='.so'
elif platform == "darwin":
    pos='.dylib'
elif platform == "win32":
    raise Exception(f"Windows is not yet supported")

DLLFOUND=False
INSTALLDIR=os.getenv('SUPERLU_LIB_PATH')

DLL = os.path.abspath(__file__ + "/../../")+'/libsuperlu_dist%s'%(pos)
if(os.path.exists(DLL)):
    DLLFOUND=True
elif(INSTALLDIR is not None):
    DLL = INSTALLDIR+'/libsuperlu_dist%s'%(pos)
    if(os.path.exists(DLL)):
        DLLFOUND=True
if(DLLFOUND == True):
    sp = ctypes.cdll.LoadLibrary(DLL)
    setup_pdbridge(sp,INT64)
else:
    raise Exception(f"Cannot find the superlu_dist library. Try to set env variable SUPERLU_LIB_PATH correctly.")

####################### initialization
pyobj = ctypes.c_void_p()
if(INT64==0):
    sp.pdbridge_init(
        n,                              # int_t m
        n,                              # int_t n
        nnz,                            # int_t nnz
        rowind.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),  # int_t *rowind
        colptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),  # int_t *colptr
        nzval.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  # double *nzval
        ctypes.byref(pyobj),            # void **pyobj
        argc,                           # int argc
        argv_ctypes                     # char *argv[]
    )
else:
    sp.pdbridge_init(
        n,                              # int_t m
        n,                              # int_t n
        nnz,                            # int_t nnz
        rowind.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),  # int_t *rowind
        colptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),  # int_t *colptr
        nzval.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  # double *nzval
        ctypes.byref(pyobj),            # void **pyobj
        argc,                           # int argc
        argv_ctypes                     # char *argv[]
    )    

####################### factor 
# Define the function signature for pdbridge_factor
sp.pdbridge_factor(
    ctypes.byref(pyobj),            # void **pyobj
)


####################### solve 
# Define the function signature for pdbridge_solve
xb = np.random.rand(n*nrhs).astype(np.float64) # pdbridge_solve will broadcast xb on rank 0 to all ranks
sp.pdbridge_solve(
    ctypes.byref(pyobj),            # void **pyobj
    nrhs,                           # int nrhs
    xb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  # double *nzval
)


####################### log-determinant 
sign = ctypes.c_int(1)
logdet = ctypes.c_double(0.0)
sp.pdbridge_logdet(
    ctypes.byref(pyobj),            # void **pyobj
    ctypes.byref(sign),                           # int nrhs
    ctypes.byref(logdet),  # double *nzval
)

if(rank==0):
    print("superlu logdet:",sign.value,logdet.value)
    sign, logdet = np.linalg.slogdet(m.toarray())
    print("numpy logdet:",int(sign),logdet)


####################### free stuff
sp.pdbridge_free(ctypes.byref(pyobj))

