
#pragma once
#include <vector>
#include <iostream>
#include "superlu_ddefs.h"
#include "lu_common.hpp"
#ifdef HAVE_CUDA
#include "lupanels_GPU.cuh"
#endif
#include "commWrapper.hpp"

struct anc25d_t
{

    MPI_Comm *comms25d;
    int maxLvl;
    int *myranks;
    int *commSizes;
    MPI_Comm *initComm(gridinfo3d_t *grid3d);

    anc25d_t(gridinfo3d_t *grid3d)
    {
        maxLvl = log2i(grid3d->zscp.Np) + 1;
        int myGrid = grid3d->zscp.Iam;

        comms25d = initComm(grid3d);
        myranks = (int *)SUPERLU_MALLOC(sizeof(int) * (maxLvl - 1));
        commSizes = (int *)SUPERLU_MALLOC(sizeof(int) * (maxLvl - 1));
        for (int i = 0; i < maxLvl - 1; i++)
        {
            commSizes[i] = 1 << (i + 1);
            myranks[i] = myGrid % (commSizes[i]);
           
        }
    }

    ~anc25d_t()
    {

        SUPERLU_FREE(myranks);
        SUPERLU_FREE(commSizes);
        for (int i = 0; i < maxLvl - 1; i++)
        {
            MPI_Comm_free(&comms25d[i]);
        }
        SUPERLU_FREE(comms25d);
    }

    // void freeComm(MPI_Comm* comm);

    /**
     * A function which returns whether the given grid is on the given rank
     *
     *  \param[in] alvl the level of the grid
     *  \param[in] k0 the grid on the given level
     *  \returns out whether the rank contains the grid
     */
    inline bool rankHasGrid(int k0, int alvl)
    {
        return k0 % commSizes[alvl - 1] == myranks[alvl - 1];
    }

    int rootRank(int k0, int alvl)
    {
        return k0 % commSizes[alvl - 1];
    }

    MPI_Comm getComm(int alvl)
    {
        return comms25d[alvl - 1];
    }
};
