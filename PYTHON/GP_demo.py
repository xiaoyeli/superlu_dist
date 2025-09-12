import numpy as np
import scipy
from pdbridge import *

######## Simulating the MCMC/LBFGS iterations 
n=4000 # starting number of samples, increment 1 each GP iteration
N_GP_iter=2
N_MCMC_iter=10
N_EI_search_iter=10

INT64 = 1 # whether to use 64bit integer (requring superlu_dist to be compiled with 64-bit indexing)
algo3d = 0 # whether to use 2D or 3D factorizations
rng = np.random.default_rng()
verbosity=True


for i in range(N_GP_iter):
    print("++ GP iteration: ", i)
    print("++ Perform MCMC ")
    for k in range(N_MCMC_iter):
        print("-- Assemble covariance matrix: ")
        a = scipy.sparse.random(n, n, density=0.01, random_state=rng)
        m = (a.T @ a) + scipy.sparse.identity(n)
        print("   sparsity: ", float(m.nnz)/n**2, "nnz(A): ", m.nnz)

        print("-- LU factor covariance matrix with ",n," samples")
        superlu_factor(m, INT64, algo3d, verbosity)

        print("-- use logdet and solve to compute likelihood: ")
        sign,logd = superlu_logdet(verbosity)
        xb = np.random.rand(n,1).astype(np.float64) 
        superlu_solve(xb, verbosity)

        if(k != N_MCMC_iter-1):
            print("-- free LU factors: ") #### Don't free LU factors at the last MCMC iteration, as the LU factors are needed for performing inference once hyperparemters are optimized 
            superlu_freeLU(verbosity)

    print("++ Perform inference or aquization function optimization here ... ")
    for ii in range(N_EI_search_iter):
        xb = np.random.rand(n,1).astype(np.float64) 
        superlu_solve(xb, verbosity)

    print("++ Add one more sample here ... ")
    superlu_freeLU(verbosity)
    n = n + 1
    ####################################################################################################

print("++ Terminate SuperLU (after this no more superLU APIs can be called) ")
superlu_terminate(verbosity)