#!/bin/bash

# cmake build for SuperLU_dist for NERSC Perlmutter GPU compute nodes
#
# Last updated: 2023/04/01
#
# Perlmutter is not yet in production and the software environment changes rapidly.
# Expect this file to be frequently updated.


# avoid GPU-Aware MPI for now until problems get resolved
module unload gpu
#module load PrgEnv-gnu
#module load gcc/11.2.0
module load cmake/3.24.3
module load cudatoolkit/11.7

cmake .. \
  -DCMAKE_C_FLAGS="-DGPU_SOLVE -std=c11 -DPRNTlevel=0 -DPROFlevel=1 -DDEBUGlevel=0 -DAdd_" \
  -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_Fortran_COMPILER=ftn \
  -DXSDK_ENABLE_Fortran=ON \
  -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
  -DTPL_ENABLE_LAPACKLIB=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DTPL_ENABLE_CUDALIB=ON \
  -DCMAKE_CUDA_FLAGS="-I${MPICH_DIR}/include -ccbin=/opt/cray/pe/craype/2.7.19/bin/CC" \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_INSTALL_PREFIX=. \
  -DCMAKE_INSTALL_LIBDIR=./lib \
  -DCMAKE_BUILD_TYPE=Release \
  -DTPL_BLAS_LIBRARIES=/opt/cray/pe/libsci/default/GNU/9.1/x86_64/lib/libsci_gnu_82_mp.so \
  -DTPL_LAPACK_LIBRARIES=/opt/cray/pe/libsci/default/GNU/9.1/x86_64/lib/libsci_gnu_82_mp.so \
  -DTPL_PARMETIS_INCLUDE_DIRS="/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/parmetis-4.0.3/include;/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/parmetis-4.0.3/metis/include" \
  -DTPL_PARMETIS_LIBRARIES="/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/parmetis-4.0.3/build/Linux-x86_64/libparmetis/libparmetis.so;/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.so" \
  -DTPL_ENABLE_COMBBLASLIB=OFF \
  -DTPL_ENABLE_NVSHMEM=OFF \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -DMPIEXEC_NUMPROC_FLAG=-n \
  -DMPIEXEC_EXECUTABLE=/usr/bin/srun \
  -DMPIEXEC_MAX_NUMPROCS=16

#make pddrive -j16
#make pddrive3d -j16
#make f_pddrive
make -j16
make install

## -DTPL_BLAS_LIBRARIES=/global/cfs/cdirs/m3894/ptlin/tpl/amd_blis/install/amd_blis-20211021-n9-gcc9.3.0/lib/libblis.a \
