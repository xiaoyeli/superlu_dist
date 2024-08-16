#include "mpi.h"
#include "superlu_defs.h"
#include "lupanels.hpp"


int_t LUstruct_v100::ancestorReduction3d(int_t ilvl, int_t *myNodeCount,
                                         int_t **treePerm)
{
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
    int_t myGrid = grid3d->zscp.Iam;

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
                zSendLPanel(k0, receiver);
                zSendUPanel(k0, receiver);
            }
            else
            {
                double alpha = 1.0, beta = 1.0; 

                zRecvLPanel(k0, sender, alpha, beta);
                zRecvUPanel(k0, sender, alpha, beta);
            }
        }
        // return 0;
        SCT->ancsReduce += SuperLU_timer_() - treduce;
    }
    return 0;
}


int_t LUstruct_v100::zSendLPanel(int_t k0, int_t receiverGrid)
{
    
	if (mycol == kcol(k0))
	{
		int_t lk = g2lCol(k0);
        if (!lPanelVec[lk].isEmpty())
		{
            MPI_Send(lPanelVec[lk].blkPtr(0), lPanelVec[lk].nzvalSize(), 
                    MPI_DOUBLE, receiverGrid, k0, grid3d->zscp.comm);
			SCT->commVolRed += lPanelVec[lk].nzvalSize() * sizeof(double);
		}
	}
	return 0;
}


int_t LUstruct_v100::zRecvLPanel(int_t k0, int_t senderGrid, double alpha, double beta)
{
    if (mycol == kcol(k0))
	{
		int_t lk = g2lCol(k0);
        if (!lPanelVec[lk].isEmpty())
		{
            
            MPI_Status status;
			MPI_Recv(LvalRecvBufs[0], lPanelVec[lk].nzvalSize(), MPI_DOUBLE, senderGrid, k0,
					 grid3d->zscp.comm, &status);

			/*reduce the updates*/
			superlu_dscal(lPanelVec[lk].nzvalSize(), alpha, lPanelVec[lk].blkPtr(0), 1);
			superlu_daxpy(lPanelVec[lk].nzvalSize(), beta, LvalRecvBufs[0], 1, lPanelVec[lk].blkPtr(0), 1);
		}
	}
	return 0;
}


int_t LUstruct_v100::zSendUPanel(int_t k0, int_t receiverGrid)
{
    
	if (myrow == krow(k0))
	{
		int_t lk = g2lRow(k0);
        if (!uPanelVec[lk].isEmpty())
		{
            MPI_Send(uPanelVec[lk].blkPtr(0), uPanelVec[lk].nzvalSize(), 
                    MPI_DOUBLE, receiverGrid, k0, grid3d->zscp.comm);
			SCT->commVolRed += uPanelVec[lk].nzvalSize() * sizeof(double);
		}
	}
	return 0;
}


int_t LUstruct_v100::zRecvUPanel(int_t k0, int_t senderGrid, double alpha, double beta)
{
    if (myrow == krow(k0))
	{
		int_t lk = g2lRow(k0);
        if (!uPanelVec[lk].isEmpty())
		{

            MPI_Status status;
			MPI_Recv(UvalRecvBufs[0], uPanelVec[lk].nzvalSize(), MPI_DOUBLE, senderGrid, k0,
					 grid3d->zscp.comm, &status);

			/*reduce the updates*/
			superlu_dscal(uPanelVec[lk].nzvalSize(), alpha, uPanelVec[lk].blkPtr(0), 1);
			superlu_daxpy(uPanelVec[lk].nzvalSize(), beta, UvalRecvBufs[0], 1, uPanelVec[lk].blkPtr(0), 1);
		}
	}
	return 0;
}


