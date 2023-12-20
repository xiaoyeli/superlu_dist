
#include "superlu_upacked.h"
#include "lupanels.hpp"
#include "xlupanels.hpp"
#include "lupanels_impl.hpp"
#include "pdgstrf3d_upacked_impl.hpp"


extern "C"
{

    dLUgpu_Handle dCreateLUgpuHandle(int_t nsupers, int_t ldt_, dtrf3Dpartition_t *trf3Dpartition,
                                     dLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
                                     SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat,
                                     double thresh_, int *info_)
    {
#if (DEBUGlevel >= 1)
        CHECK_MALLOC(grid3d->iam, "Enter createLUgpuHandle");
#endif

        xLUstruct_t<double> *instance = new xLUstruct_t<double>(nsupers, ldt_, trf3Dpartition,
                                                                    LUstruct, grid3d,
                                                                    SCT_, options_, stat,
                                                                    thresh_, info_);

        return reinterpret_cast<dLUgpu_Handle>(instance);
        //    return new xLUstruct_t<double>(nsupers, ldt_, trf3Dpartition,
        // 			    LUstruct, grid3d,
        // 			    SCT_, options_, stat,
        // 			    thresh_, info_);

#if (DEBUGlevel >= 1)
        CHECK_MALLOC(grid3d->iam, "Exit createLUgpuHandle");
#endif
    }

    void dDestroyLUgpuHandle(dLUgpu_Handle LuH)
    {
        printf("\t... before delete luH\n");
        fflush(stdout);
        delete reinterpret_cast<xLUstruct_t<double> *>(LuH);
        printf("\t... after delete luH\n");
        fflush(stdout);
    }

    // I think the following is not used 
    int dGatherFactoredLU3Dto2D(dLUgpu_Handle LuH);

    int dCopyLUGPU2Host(dLUgpu_Handle LuH, dLUstruct_t *LUstruct)
    {
        
        xLUstruct_t<double> *LU_v1 = reinterpret_cast<xLUstruct_t<double> *>(LuH);
        double tXferGpu2Host = SuperLU_timer_();
        if (LU_v1->superlu_acc_offload)
        {
#ifdef HAVE_CUDA
            cudaStreamSynchronize(LU_v1->A_gpu.cuStreams[0]); // in theory I don't need it
            LU_v1->copyLUGPUtoHost();
#endif
        }

        LU_v1->packedU2skyline(LUstruct);
        tXferGpu2Host = SuperLU_timer_() - tXferGpu2Host;
        printf("Time to send data back= %g\n", tXferGpu2Host);

        return 0;
    }

    int pdgstrf3d_LUv1(dLUgpu_Handle LUHand)
    {
        // perform the factorization
        xLUstruct_t<double> *LU_v1 = reinterpret_cast<xLUstruct_t<double> *>(LUHand);
        return LU_v1->pdgstrf3d();
        // return LUHand->pdgstrf3d();
    }
}
