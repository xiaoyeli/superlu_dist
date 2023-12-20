
#include "superlu_upacked.h"
#include "lupanels.hpp"

extern "C"
{

LUgpu_Handle createLUgpuHandle(int_t nsupers, int_t ldt_, dtrf3Dpartition_t *trf3Dpartition,
                  dLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
                  SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat,
                  double thresh_, int *info_)
{
#if (DEBUGlevel >= 1)
	CHECK_MALLOC(grid3d->iam, "Enter createLUgpuHandle");
#endif
  
   return new LUstruct_v100(nsupers, ldt_, trf3Dpartition,
			    LUstruct, grid3d,
			    SCT_, options_, stat,
			    thresh_, info_);
   
#if (DEBUGlevel >= 1)
	CHECK_MALLOC(grid3d->iam, "Exit createLUgpuHandle");
#endif
} 

void destroyLUgpuHandle(LUgpu_Handle LuH)
{
  printf("\t... before delete luH\n"); fflush(stdout);
   delete LuH; 
  printf("\t... after delete luH\n"); fflush(stdout);
}

int dgatherFactoredLU3Dto2D(LUgpu_Handle LuH);

int copyLUGPU2Host(LUgpu_Handle LuH, dLUstruct_t *LUstruct)
{
   double tXferGpu2Host = SuperLU_timer_();
   if (LuH->superlu_acc_offload)
   {
      #ifdef HAVE_CUDA
      cudaStreamSynchronize(LuH->A_gpu.cuStreams[0]);    // in theory I don't need it 
      LuH->copyLUGPUtoHost();
      #endif
   }
      
   LuH->packedU2skyline(LUstruct);
   tXferGpu2Host = SuperLU_timer_()-tXferGpu2Host;
   printf("Time to send data back= %g\n",tXferGpu2Host );

   return 0; 
}

  int pdgstrf3d_LUpackedInterface( LUgpu_Handle LUHand )
{
   // perform the factorization
   return LUHand->pdgstrf3d();
}




}

