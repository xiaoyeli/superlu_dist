import numpy as np
import os
import ctypes
import scipy
from scipy.sparse.linalg import splu
import time
import sys
import pickle

####################################################################################################
####################################################################################################
####################### create the matrix
INT64 = 0 # whether to use 64bit integer (requring superlu_dist to be compiled with 64-bit indexing)
rng = np.random.default_rng()
n = 4000000
nrhs = 1
use_cov = 1


######################## define the files used to communicate between masters and workers
CONTROL_FILE = "control.txt"
MODE_FILE  = "mode.txt"
DATA_FILE    = "data.bin"
RESULT_FILE  = "result.bin"

def wait_for_flag(expected_flag, control_file, poll_interval=0.1):
    """Poll the control file until its content equals the expected flag."""
    while True:
        if os.path.exists(control_file):
            with open(control_file, "r") as f:
                flag = f.read().strip()
            if flag == expected_flag:
                return True
        time.sleep(poll_interval)




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



####################################################################################################
####################################################################################################
####################### call the APIs

####################### initialization
start = time.time()
with open(DATA_FILE, "wb") as f:
    pickle.dump((m,INT64), f)
with open(CONTROL_FILE, "w") as f:
    f.write("init")
wait_for_flag("done", CONTROL_FILE)
end = time.time()
print(f"Time spent in pdbridge_init: {end - start} seconds")


####################### factor
start = time.time()
with open(CONTROL_FILE, "w") as f:
    f.write("factor")
wait_for_flag("done", CONTROL_FILE)
end = time.time()
print(f"Time spent in pdbridge_factor: {end - start} seconds")


####################### solve 
xb = np.random.rand(n*nrhs).astype(np.float64) 
start = time.time()
with open(DATA_FILE, "wb") as f:
    pickle.dump((xb,nrhs), f)
with open(CONTROL_FILE, "w") as f:
    f.write("solve")
wait_for_flag("done", CONTROL_FILE)
with open(RESULT_FILE, "rb") as f:
    xb = pickle.load(f)
end = time.time()
print(f"Time spent in pdbridge_solve: {end - start} seconds")



####################### log-determinant  
start = time.time()
with open(CONTROL_FILE, "w") as f:
    f.write("logdet")
wait_for_flag("done", CONTROL_FILE)
with open(RESULT_FILE, "rb") as f:
    sign,logdet = pickle.load(f)
end = time.time()
print(f"Time spent in pdbridge_logdet: {end - start} seconds")


####################### free stuff
start = time.time()
with open(CONTROL_FILE, "w") as f:
    f.write("free")
wait_for_flag("done", CONTROL_FILE)
end = time.time()
print(f"Time spent in pdbridge_free: {end - start} seconds")


####################### terminate all workers if no more superLU calls are needed
start = time.time()
with open(CONTROL_FILE, "w") as f:
    f.write("terminate")
end = time.time()
print(f"Time spent in pdbridge_terminate: {end - start} seconds")

