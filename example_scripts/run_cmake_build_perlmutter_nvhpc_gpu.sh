module load PrgEnv-nvidia
module load cmake
module load cudatoolkit

cmake .. \
     -DTPL_PARMETIS_LIBRARIES=ON \
     -DTPL_PARMETIS_INCLUDE_DIRS="/global/cfs/cdirs/m3894/lib/PrgEnv-nvidia/parmetis-4.0.3/include;/global/cfs/cdirs/m3894/lib/PrgEnv-nvidia/parmetis-4.0.3/metis/include" \
     -DTPL_PARMETIS_LIBRARIES="/global/cfs/cdirs/m3894/lib/PrgEnv-nvidia/parmetis-4.0.3/build/Linux-x86_64/libparmetis/libparmetis.so;/global/cfs/cdirs/m3894/lib/PrgEnv-nvidia/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.so" \
     -DTPL_ENABLE_COMBBLASLIB=OFF \
     -Denable_openmp=OFF \
     -DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=1 -DDEBUGlevel=0 -DAdd_ -fortranlibs" \
     -DCMAKE_CXX_FLAGS="-DRELEASE -fortranlibs " \
     -DCMAKE_CUDA_FLAGS="-ccbin nvc++" \
     -DCMAKE_C_COMPILER=cc \
     -DCMAKE_CXX_COMPILER=CC \
     -DXSDK_ENABLE_Fortran=ON \
     -DCMAKE_CUDA_FLAGS="-I${NVSHMEM_HOME}/include -I${MPICH_DIR}/include -ccbin=/opt/cray/pe/craype/2.7.30/bin/CC" \
     -DCMAKE_CUDA_ARCHITECTURES=80 \
     -DTPL_ENABLE_CUDALIB=TRUE \
     -DTPL_CUDA_LIBRARIES="/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/targets/x86_64-linux/lib/libcudart.so" \
     -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
     -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
     -DTPL_BLAS_LIBRARIES=/opt/cray/pe/libsci/23.12.5/NVIDIA/23.3/x86_64/lib/libsci_nvidia_mp.a \
     -DBUILD_SHARED_LIBS=OFF \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=. \
     -DMPIEXEC_NUMPROC_FLAG=-n \
     -DMPIEXEC_EXECUTABLE=/usr/bin/srun \
     -DMPIEXEC_MAX_NUMPROCS=16
     
make f_pddrive
make pddrive
make pddrive3d
