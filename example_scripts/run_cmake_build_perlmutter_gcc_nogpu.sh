module swap PrgEnv-nvidia PrgEnv-gnu
module load gcc #/10.3.0
module load cmake/3.22.0
module load cudatoolkit

cmake .. \
     -DTPL_PARMETIS_LIBRARIES=ON \
     -DTPL_PARMETIS_INCLUDE_DIRS="/global/cfs/cdirs/m3894/ptlin/tpl/parmetis/parmetis-4.0.3/include;/global/cfs/cdirs/m3894/ptlin/tpl/parmetis/parmetis-4.0.3/metis/include" \
     -DTPL_PARMETIS_LIBRARIES="/global/cfs/cdirs/m3894/ptlin/tpl/parmetis/parmetis-4.0.3/build/Linux-x86_64/libparmetis/libparmetis.a;/global/cfs/cdirs/m3894/ptlin/tpl/parmetis/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.a" \
     -DTPL_ENABLE_COMBBLASLIB=OFF \
     -DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=1 -DDEBUGlevel=0 -DAdd_" \
     -DCMAKE_C_COMPILER=cc \
     -DCMAKE_CXX_COMPILER=CC \
     -DXSDK_ENABLE_Fortran=ON \
     -DCMAKE_CUDA_ARCHITECTURES=80 \
     -DCMAKE_CUDA_FLAGS="-I${MPICH_DIR}/include" \
     -DTPL_ENABLE_CUDALIB=FALSE \
     -DTPL_CUDA_LIBRARIES="/global/common/software/nersc/cos1.3/cuda/11.3.0/targets/x86_64-linux/lib/libcublas.so;/global/common/software/nersc/cos1.3/cuda/11.3.0/targets/x86_64-linux/lib/libcudart.so" \
     -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
     -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
     -DTPL_BLAS_LIBRARIES=/opt/cray/pe/libsci/21.08.1.2/GNU/9.1/x86_64/lib/libsci_gnu_82_mpi_mp.a \
     -DBUILD_SHARED_LIBS=OFF \
     -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_INSTALL_PREFIX=. \
     -DMPIEXEC_NUMPROC_FLAG=-n \
     -DMPIEXEC_EXECUTABLE=/usr/bin/srun \
     -DMPIEXEC_MAX_NUMPROCS=16
     
make pddrive


# -DTPL_BLAS_LIBRARIES=/global/cfs/cdirs/m3894/ptlin/tpl/amd_blis/install/amd_blis-20211021-n9-gcc9.3.0/lib/libblis.a \