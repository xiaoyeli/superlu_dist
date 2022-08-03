#!/bin/bash

# cmake build for SuperLU_dist on NERSC Perlmutter
#
# Last update: July 21, 2022
# Perlmutter is not in production and the software environment changes rapidly.
# Expect this file to be frequently updated
#
# This build script targets Perlmutter Slingshot 11 GPU compute nodes
#
# When requesting GPU compute nodes using salloc, will have to add the _ss11 suffix to the QOS
# For example, if previously requested 1 node using salloc as follows
# salloc -C gpu -N 1 -G 4 -t 30 -A m3894 -q regular
# will now need:
# salloc -C gpu -N 1 -G 4 -t 30 -A m3894 -q regular_ss11
# will also need to issue the 5 module load commands below and the
# two "export LD_LIBRARY_PATH" commands below in the shell after
# receiving node allocation or in scripts that will run on the nodes
#
# For sbatch scripts, if previously had
# #SBATCH -q regular
# will now need
# #SBATCH -q regular_ss11
# will also need to include the 5 module load commands below and the
# two "export LD_LIBRARY_PATH" commands below in the batch script
#
# Note: you may have to specify your own parmetis/metis libraries


module load PrgEnv-gnu
module load gcc/11.2.0
module load cmake/3.22.0
module load cudatoolkit/11.7
# avoid bug in cray-libsci/21.08.1.2
module load cray-libsci/22.06.1.3

# avoid bug in cudatoolkit
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH//\/usr\/local\/cuda-11.5\/compat:/}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH//\/usr\/local\/cuda-11.7\/compat:/}

cmake .. \
  -DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0 -DAdd_" \
  -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_Fortran_COMPILER=ftn \
  -DXSDK_ENABLE_Fortran=ON \
  -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DTPL_ENABLE_CUDALIB=ON \
  -DCMAKE_CUDA_FLAGS="-I${MPICH_DIR}/include -ccbin=/opt/cray/pe/craype/2.7.16/bin/CC" \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_INSTALL_PREFIX=. \
  -DCMAKE_INSTALL_LIBDIR=./lib \
  -DCMAKE_BUILD_TYPE=Release \
  -DTPL_BLAS_LIBRARIES=/opt/cray/pe/libsci/22.06.1.3/GNU/9.1/x86_64/lib/libsci_gnu_81_mp.so \
  -DTPL_PARMETIS_INCLUDE_DIRS="/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/parmetis-4.0.3/include;/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/parmetis-4.0.3/metis/include" \
  -DTPL_PARMETIS_LIBRARIES="/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/parmetis-4.0.3/build/Linux-x86_64/libparmetis/libparmetis.so;/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.so" \
  -DTPL_ENABLE_COMBBLASLIB=OFF \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -DMPIEXEC_NUMPROC_FLAG=-n \
  -DMPIEXEC_EXECUTABLE=/usr/bin/srun \
  -DMPIEXEC_MAX_NUMPROCS=16
     
make pddrive
make pddrive3d
make f_pddrive

# -DTPL_BLAS_LIBRARIES=/global/cfs/cdirs/m3894/ptlin/tpl/amd_blis/install/amd_blis-20211021-n9-gcc9.3.0/lib/libblis.a \
