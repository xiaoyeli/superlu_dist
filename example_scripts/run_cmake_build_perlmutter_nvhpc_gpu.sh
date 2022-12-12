module load PrgEnv-nvidia
module load cmake/3.22.0
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
     -DCMAKE_CUDA_ARCHITECTURES=80 \
     -DCMAKE_CUDA_FLAGS="-I${MPICH_DIR}/include" \
     -DTPL_ENABLE_CUDALIB=TRUE \
     -DTPL_CUDA_LIBRARIES="/global/common/software/nersc/cos1.3/cuda/11.3.0/targets/x86_64-linux/lib/libcublas.so;/global/common/software/nersc/cos1.3/cuda/11.3.0/targets/x86_64-linux/lib/libcudart.so" \
     -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
     -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
     -DTPL_BLAS_LIBRARIES=/opt/cray/pe/libsci/21.08.1.2/NVIDIA/20.7/x86_64/lib/libsci_nvidia_mpi_mp.a \
     -DBUILD_SHARED_LIBS=OFF \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=. \
     -DMPIEXEC_NUMPROC_FLAG=-n \
     -DMPIEXEC_EXECUTABLE=/usr/bin/srun \
     -DMPIEXEC_MAX_NUMPROCS=16
     
make f_pddrive
