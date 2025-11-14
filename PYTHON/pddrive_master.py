import numpy as np
import os
import ctypes
import scipy
from scipy.sparse.linalg import splu
import time
import sys
import pickle
from pdbridge import *




####################################################################################################
####################################################################################################
####################### create the matrix
INT64 = 1 # whether to use 64bit integer (requring superlu_dist to be compiled with 64-bit indexing)
algo3d = 0 # whether to use 2D or 3D factorizations
rng = np.random.default_rng()
n = 4000000
nrhs = 1
use_cov = 0
verbosity=True

if(use_cov==0):
    a = scipy.sparse.random(n, n, density=0.01, random_state=rng)
    m = (a.T @ a) + scipy.sparse.identity(n)
    print("sparsity: ", float(m.nnz)/n**2, "nnz(A): ", m.nnz)
else:
    m = scipy.sparse.load_npz('/global/cfs/cdirs/m2957/liuyangz/my_research/matrix/sparse_matrix10Mill.npz')
    # m = scipy.sparse.load_npz('/global/cfs/cdirs/m2957/liuyangz/my_research/matrix/sparse_matrix10Mill_prettydense.npz')
    # m = scipy.sparse.load_npz('/global/cfs/cdirs/m2957/liuyangz/my_research/matrix/sparse_matrix10Mill_no1.npz')
    # m = scipy.sparse.load_npz('/global/cfs/cdirs/m2957/liuyangz/my_research/matrix/sparse_matrix10Mill_no1.npz')
    m = m.tocsr()
    m = m[0:n, 0:n]
    print("sparsity: ", float(m.nnz)/n**2, "nnz(A): ", m.nnz)


####################################################################################################
####################################################################################################
####################### call the APIs

xb = np.random.rand(n,nrhs).astype(np.float64) 
superlu_factor(m, INT64, algo3d, verbosity)
sign,logd = superlu_logdet(verbosity)
superlu_solve(xb, verbosity)
superlu_freeLU(verbosity)
superlu_terminate(verbosity)
