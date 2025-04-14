import numpy as np
import os
import ctypes
import scipy
from scipy.sparse.linalg import splu
import time
import sys
import mpi4py
from mpi4py import MPI
import pdbridge


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if(rank==0):
    print('mpi4py version: ', mpi4py.__version__)
    print('MPI count:', size)


####################################################################################################
####################################################################################################
####################### create the matrix
INT64 = 0 # whether to use 64bit integer (requring superlu_dist to be compiled with 64-bit indexing)
algo3d = 1 # whether to use 2D or 3D factorizations
rng = np.random.default_rng()
n = 4000000
nrhs = 1
use_cov = 1

if(rank==0):
    if(use_cov==0):
        a = scipy.sparse.random(n, n, density=0.01, random_state=rng)
        m = (a.T @ a) + scipy.sparse.identity(n)
        print("sparsity: ", float(m.nnz)/n**2, "nnz(A): ", m.nnz)
    else:
        m = scipy.sparse.load_npz('/global/cfs/cdirs/m2957/liuyangz/my_research/matrix/sparse_matrix10Mill.npz')
        # m = scipy.sparse.load_npz('/global/cfs/cdirs/m2957/liuyangz/my_research/matrix/sparse_matrix10Mill_prettydense.npz')
        m = m.tocsr()
        m = m[0:n, 0:n]
        print("sparsity: ", float(m.nnz)/n**2, "nnz(A): ", m.nnz)
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




####################################################################################################
####################################################################################################
####################### handle options 
argv=sys.argv
if(len(argv)==1): # options are not passed via command line, set them manually here. If they are not set here, default values are used
    if(algo3d==1):
        argv.extend(['-d', '1'])  # process layers
        argv.extend(['-r', '%i'%(np.sqrt(size))])  # process rows
        argv.extend(['-c', '%i'%(np.sqrt(size))])  # process columns       
    else:
        argv.extend(['-r', '%i'%(np.sqrt(size))])  # process rows
        argv.extend(['-c', '%i'%(np.sqrt(size))])  # process columns
    
    argv.extend(['-p', '1'])  # row permutation 
    argv.extend(['-q', '2'])  # column permutation 
    argv.extend(['-s', '0'])  # parallel symbolic factorization, needs -q to be 5
    argv.extend(['-i', '0'])  # whether to use iterative refinement 0, 1, 2
    argv.extend(['-m', '0'])  # whether to use symmetric pattern 0 or 1
argc = len(argv)
if(rank==0):    
    print('SuperLU options: ',argv[1:])
argv_bytes = [arg.encode('utf-8') for arg in argv]
argv_ctypes = (ctypes.c_char_p * (argc + 1))(*argv_bytes, None)






####################################################################################################
####################################################################################################
####################### call the APIs
start = time.time()
sp = pdbridge.load_library(INT64)
####################### initialization
pyobj = ctypes.c_void_p()
if(INT64==0):
    sp.pdbridge_init(
        algo3d,
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
        algo3d,
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
end = time.time()
if(rank==0):
    print(f"Time spent in pdbridge_init: {end - start} seconds")



####################### factor 
start = time.time()
sp.pdbridge_factor(
    ctypes.byref(pyobj),            # void **pyobj
)
end = time.time()
if(rank==0):
    print(f"Time spent in pdbridge_factor: {end - start} seconds")


####################### solve 
start = time.time()
xb = np.random.rand(n*nrhs).astype(np.float64) # pdbridge_solve will broadcast xb on rank 0 to all ranks
sp.pdbridge_solve(
    ctypes.byref(pyobj),            # void **pyobj
    nrhs,                           # int nrhs
    xb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  # double *nzval
)
end = time.time()
if(rank==0):
    print(f"Time spent in pdbridge_solve: {end - start} seconds")



####################### log-determinant 
start = time.time()
sign = ctypes.c_int(1)
logdet = ctypes.c_double(0.0)
sp.pdbridge_logdet(
    ctypes.byref(pyobj),            # void **pyobj
    ctypes.byref(sign),                           # int nrhs
    ctypes.byref(logdet),  # double *nzval
)

if(rank==0):
    print("superlu logdet:",sign.value,logdet.value)
    # sign, logdet = np.linalg.slogdet(m.toarray())
    # print("numpy logdet:",int(sign),logdet)
end = time.time()
if(rank==0):
    print(f"Time spent in pdbridge_logdet: {end - start} seconds")


####################### free stuff
start = time.time()
sp.pdbridge_free(ctypes.byref(pyobj))
end = time.time()
if(rank==0):
    print(f"Time spent in pdbridge_free: {end - start} seconds")

