#pragma once 
#include "mpi.h"
#include "superlu_defs.h"
#include "lupanels.hpp"
#include "luAuxStructTemplated.hpp"
#include "superlu_blas.hpp"

template <typename Ftype>
int_t xLUstruct_t<Ftype>::ancestorReduction3d(int_t ilvl, int_t *myNodeCount,
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
                Ftype alpha = one<Ftype>(); Ftype beta = one<Ftype>(); 

                zRecvLPanel(k0, sender, alpha, beta);
                zRecvUPanel(k0, sender, alpha, beta);
            }
        }
        // return 0;
        SCT->ancsReduce += SuperLU_timer_() - treduce;
    }
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::zSendLPanel(int_t k0, int_t receiverGrid)
{
    
	if (mycol == kcol(k0))
	{
		int_t lk = g2lCol(k0);
        if (!lPanelVec[lk].isEmpty())
		{
            MPI_Send(lPanelVec[lk].blkPtr(0), lPanelVec[lk].nzvalSize(), 
                    get_mpi_type<Ftype>(), receiverGrid, k0, grid3d->zscp.comm);
			SCT->commVolRed += lPanelVec[lk].nzvalSize() * sizeof(Ftype);
		}
	}
	return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::zRecvLPanel(int_t k0, int_t senderGrid, Ftype alpha, Ftype beta)
{
    if (mycol == kcol(k0))
	{
		int_t lk = g2lCol(k0);
        if (!lPanelVec[lk].isEmpty())
		{
            
            MPI_Status status;
			MPI_Recv(LvalRecvBufs[0], lPanelVec[lk].nzvalSize(), get_mpi_type<Ftype>(), senderGrid, k0,
					 grid3d->zscp.comm, &status);

			/*reduce the updates*/
			superlu_scal<Ftype>(lPanelVec[lk].nzvalSize(), alpha, lPanelVec[lk].blkPtr(0), 1);
			superlu_axpy<Ftype>(lPanelVec[lk].nzvalSize(), beta, LvalRecvBufs[0], 1, lPanelVec[lk].blkPtr(0), 1);
		}
	}
	return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::zSendUPanel(int_t k0, int_t receiverGrid)
{
    
	if (myrow == krow(k0))
	{
		int_t lk = g2lRow(k0);
        if (!uPanelVec[lk].isEmpty())
		{
            MPI_Send(uPanelVec[lk].blkPtr(0), uPanelVec[lk].nzvalSize(), 
                    get_mpi_type<Ftype>(), receiverGrid, k0, grid3d->zscp.comm);
			SCT->commVolRed += uPanelVec[lk].nzvalSize() * sizeof(Ftype);
		}
	}
	return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::zRecvUPanel(int_t k0, int_t senderGrid, Ftype alpha, Ftype beta)
{
    if (myrow == krow(k0))
	{
		int_t lk = g2lRow(k0);
        if (!uPanelVec[lk].isEmpty())
		{

            MPI_Status status;
			MPI_Recv(UvalRecvBufs[0], uPanelVec[lk].nzvalSize(), get_mpi_type<Ftype>(), senderGrid, k0,
					 grid3d->zscp.comm, &status);

			/*reduce the updates*/
			superlu_scal<Ftype>(uPanelVec[lk].nzvalSize(), alpha, uPanelVec[lk].blkPtr(0), 1);
			superlu_axpy<Ftype>(uPanelVec[lk].nzvalSize(), beta, UvalRecvBufs[0], 1, uPanelVec[lk].blkPtr(0), 1);
		}
	}
	return 0;
}


