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
import pickle


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if(rank==0):
    print('mpi4py version: ', mpi4py.__version__)
    print('MPI count:', size)



######################## define the files used to communicate between masters and workers
CONTROL_FILE=os.getenv("CONTROL_FILE", "control.txt")
DATA_FILE=os.getenv("DATA_FILE", "data.bin")
RESULT_FILE=os.getenv("RESULT_FILE", "result.bin")  
poll_interval = 0.1


# Ensure the file exists; if not, wait a moment and try again.
while True:
    flag=''
    if rank == 0:
        while True:
            if os.path.exists(CONTROL_FILE):
                with open(CONTROL_FILE, "r") as f:
                    flag = f.read().strip()
                if flag == "init" or flag == "factor" or flag == "solve" or flag=="logdet" or flag=="free" or flag=="terminate":
                    break
            time.sleep(poll_interval)
    flag = comm.bcast(flag, root=0)
    if(flag=="init"):
        #####  read in the matrix by rank 0
        if rank == 0:
            with open(DATA_FILE, "rb") as f:
                m,INT64,algo3d = pickle.load(f)
            n=(m.shape)[0]    
            INT64 = comm.bcast(INT64, root=0)
            algo3d = comm.bcast(algo3d, root=0)
            n = comm.bcast(n, root=0)
        else:
            INT64=-1
            algo3d=-1
            n=-1
            INT64 = comm.bcast(INT64, root=0)
            algo3d = comm.bcast(algo3d, root=0)
            n = comm.bcast(n, root=0)
            a = scipy.sparse.random(1, 1, density=1)
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
            if(algo3d==1):
                argv.extend(['-d', '1'])  # process layers
                argv.extend(['-r', '%i'%(np.sqrt(size))])  # process rows
                argv.extend(['-c', '%i'%(np.sqrt(size))])  # process columns
                argv.extend(['-b', '0'])  # batch count             
            else:
                argv.extend(['-r', '%i'%(np.sqrt(size))])  # process rows
                argv.extend(['-c', '%i'%(np.sqrt(size))])  # process columns
            
            argv.extend(['-p', '1'])  # row permutation 
            argv.extend(['-q', '2'])  # column permutation 
            argv.extend(['-s', '0'])  # parallel symbolic factorization, needs -q to be 5
            argv.extend(['-i', '0'])  # whether to use iterative refinement 0, 1, 2
            argv.extend(['-m', '0'])  # whether to use symmetric pattern 0 or 1
            argv.extend(['-n', '1'])  # whether to use tiny pivot replacement
        argc = len(argv)
        if(rank==0):    
            print('SuperLU options: ',argv[1:])
        argv_bytes = [arg.encode('utf-8') for arg in argv]
        argv_ctypes = (ctypes.c_char_p * (argc + 1))(*argv_bytes, None)

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

    elif(flag=="factor"):
        ####################### factor 
        sp.pdbridge_factor(
            ctypes.byref(pyobj),            # void **pyobj
        )
    elif(flag=="solve"):              
        ####################### solve 
        #####  read in the RHS by rank 0
        if rank == 0:
            with open(DATA_FILE, "rb") as f:
                xb,nrhs = pickle.load(f)
            nrhs = comm.bcast(nrhs, root=0)
            xb = np.ascontiguousarray(xb, dtype=np.float64)            
        else:
            nrhs=-1  
            nrhs = comm.bcast(nrhs, root=0)
            xb = np.random.rand(n*nrhs).astype(np.float64) # pdbridge_solve will broadcast xb on rank 0 to all ranks      

        sp.pdbridge_solve(
            ctypes.byref(pyobj),            # void **pyobj
            nrhs,                           # int nrhs
            xb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  # double *nzval
        )
        if rank == 0:
            with open(RESULT_FILE, "wb") as f:
                pickle.dump(xb, f)

    elif(flag=="logdet"):
        ####################### log-determinant 
        sign = ctypes.c_int(1)
        logdet = ctypes.c_double(0.0)
        sp.pdbridge_logdet(
            ctypes.byref(pyobj),            # void **pyobj
            ctypes.byref(sign),                           # int nrhs
            ctypes.byref(logdet),  # double *nzval
        )
        if rank == 0:
            with open(RESULT_FILE, "wb") as f:
                pickle.dump((sign.value, logdet.value),f)

        if(rank==0):
            print("superlu logdet:",sign.value,logdet.value)
            # sign, logdet = np.linalg.slogdet(m.toarray())
            # print("numpy logdet:",int(sign),logdet)     

    elif(flag=="free"):
        ####################### free stuff
        sp.pdbridge_free(ctypes.byref(pyobj))
    elif(flag=="terminate"):      
        break

    if(rank==0):    
        ####################### signal the master that the work (init, factor, solve, logdet, free) has been completed
        with open(CONTROL_FILE, "w") as f:
            if(flag=="free"):
                f.write("clean")                
            else:
                f.write("done")

if(rank==0):    
    ####################### signal the master that the work (terminate) has been completed and the MPI communicator will be released
    with open(CONTROL_FILE, "w") as f:
        f.write("finish")