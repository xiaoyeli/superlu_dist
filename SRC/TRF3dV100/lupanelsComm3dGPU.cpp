#include "mpi.h"
// #include "cublasDefs.hhandle, "
#include "lupanels.hpp"

#ifdef HAVE_CUDA

int_t LUstruct_v100::ancestorReduction3dGPU(int_t ilvl, int_t *myNodeCount,
                                         int_t **treePerm)
{
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
    int_t myGrid = grid3d->zscp.Iam;

#if (DEBUGlevel >= 1)
    printf(".maxLvl %d\n", maxLvl); fflush(stdout);
    CHECK_MALLOC(grid3d->iam, "Enter ancestorReduction3dGPU()");
#endif
	
    int_t sender, receiver;
    if ((myGrid % (1 << (ilvl + 1))) == 0)
    {
        sender = myGrid + (1 << ilvl);
        receiver = myGrid;
    }
    else
    {
        sender = myGrid;
        receiver = myGrid - (1 << ilvl);
    }

    /*Reduce all the ancestors*/
    for (int_t alvl = ilvl + 1; alvl < maxLvl; ++alvl)
    {
        /* code */
        // int_t atree = myTreeIdxs[alvl];
        int_t numNodes = myNodeCount[alvl];
        int_t *nodeList = treePerm[alvl];
        double treduce = SuperLU_timer_();
        

        /*first setting the L blocks to zero*/
        for (int_t node = 0; node < numNodes; ++node) /* for each block column ... */
        {
            int_t k0 = nodeList[node];

            if (myGrid == sender)
            {
                zSendLPanelGPU(k0, receiver);
                zSendUPanelGPU(k0, receiver);
            }
            else
            {
                double alpha = 1.0, beta = 1.0; 

                zRecvLPanelGPU(k0, sender, alpha, beta);
                zRecvUPanelGPU(k0, sender, alpha, beta);
            }
        }
        cudaStreamSynchronize(A_gpu.cuStreams[0]) ;
        // return 0;
        SCT->ancsReduce += SuperLU_timer_() - treduce;
    }
    
#if (DEBUGlevel >= 1)
        CHECK_MALLOC(grid3d->iam, "Exit ancestorReduction3dGPU()");
#endif
    
    return 0;
}


int_t LUstruct_v100::zSendLPanelGPU(int_t k0, int_t receiverGrid)
{
    
	if (mycol == kcol(k0))
	{
		int_t lk = g2lCol(k0);
        if (!lPanelVec[lk].isEmpty())
		{
            MPI_Send(lPanelVec[lk].blkPtrGPU(0), lPanelVec[lk].nzvalSize(), 
                    MPI_DOUBLE, receiverGrid, k0, grid3d->zscp.comm);
			SCT->commVolRed += lPanelVec[lk].nzvalSize() * sizeof(double);
		}
	}
	return 0;
}


int_t LUstruct_v100::zRecvLPanelGPU(int_t k0, int_t senderGrid, double alpha, double beta)
{
    if (mycol == kcol(k0))
	{
		int_t lk = g2lCol(k0);
        if (!lPanelVec[lk].isEmpty())
		{
            
            MPI_Status status;
			MPI_Recv(A_gpu.LvalRecvBufs[0], lPanelVec[lk].nzvalSize(), MPI_DOUBLE, senderGrid, k0,
					 grid3d->zscp.comm, &status);

			/*reduce the updates*/
            cublasHandle_t handle=A_gpu.cuHandles[0];
            cudaStream_t cuStream = A_gpu.cuStreams[0];
            cublasSetStream(handle, cuStream);
			cublasDscal(handle, lPanelVec[lk].nzvalSize(), &alpha, lPanelVec[lk].blkPtrGPU(0), 1);
			cublasDaxpy(handle, lPanelVec[lk].nzvalSize(), &beta, A_gpu.LvalRecvBufs[0], 1, lPanelVec[lk].blkPtrGPU(0), 1);
            cudaStreamSynchronize(cuStream);
            // cublasDscal(cublasHandle_t handle, int n,
            //                 const double          *alpha,
            //                 double          *x, int incx)
            // cublasDaxpy(cublasHandle_t handle, int n,
            //                const double          *alpha,
            //                const double          *x, int incx,
            //                double                *y, int incy)
		}
	}
	return 0;
}


int_t LUstruct_v100::zSendUPanelGPU(int_t k0, int_t receiverGrid)
{
    
	if (myrow == krow(k0))
	{
		int_t lk = g2lRow(k0);
        if (!uPanelVec[lk].isEmpty())
		{
            MPI_Send(uPanelVec[lk].blkPtrGPU(0), uPanelVec[lk].nzvalSize(), 
                    MPI_DOUBLE, receiverGrid, k0, grid3d->zscp.comm);
			SCT->commVolRed += uPanelVec[lk].nzvalSize() * sizeof(double);
		}
	}
	return 0;
}


int_t LUstruct_v100::zRecvUPanelGPU(int_t k0, int_t senderGrid, double alpha, double beta)
{
    if (myrow == krow(k0))
	{
		int_t lk = g2lRow(k0);
        if (!uPanelVec[lk].isEmpty())
		{

            MPI_Status status;
			MPI_Recv(A_gpu.UvalRecvBufs[0], uPanelVec[lk].nzvalSize(), MPI_DOUBLE, senderGrid, k0,
					 grid3d->zscp.comm, &status);

			/*reduce the updates*/
            cublasHandle_t handle=A_gpu.cuHandles[0];
            cudaStream_t cuStream = A_gpu.cuStreams[0];
            cublasSetStream(handle, cuStream);
			cublasDscal(handle, uPanelVec[lk].nzvalSize(), &alpha, uPanelVec[lk].blkPtrGPU(0), 1);
			cublasDaxpy(handle, uPanelVec[lk].nzvalSize(), &beta, A_gpu.UvalRecvBufs[0], 1, uPanelVec[lk].blkPtrGPU(0), 1);
            cudaStreamSynchronize(cuStream);
		}
	}
	return 0;
}

#endif
