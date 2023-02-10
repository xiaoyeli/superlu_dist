module load PrgEnv-gnu
module load gcc/11.2.0
module load cmake/3.22.0
module load cudatoolkit/11.7
# avoid bug in cray-libsci/21.08.1.2
module load cray-libsci/22.11.1.2

# avoid bug in cudatoolkit
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH//\/usr\/local\/cuda-11.5\/compat:/}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH//\/usr\/local\/cuda-11.7\/compat:/}

cmake .. \
     -DTPL_PARMETIS_LIBRARIES=ON \
     -DTPL_PARMETIS_INCLUDE_DIRS="/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/parmetis-4.0.3/include;/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/parmetis-4.0.3/metis/include" \
     -DTPL_PARMETIS_LIBRARIES="/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/parmetis-4.0.3/build/Linux-x86_64/libparmetis/libparmetis.so;/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.so" \
     -DTPL_ENABLE_COMBBLASLIB=OFF \
     -DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=0 -DDEBUGlevel=0 -DAdd_" \
     -DCMAKE_C_COMPILER=cc \
     -DCMAKE_CXX_COMPILER=CC \
     -DXSDK_ENABLE_Fortran=ON \
     -DCMAKE_CUDA_ARCHITECTURES=80 \
     -DCMAKE_CUDA_FLAGS="-I${MPICH_DIR}/include" \
     -DTPL_ENABLE_CUDALIB=FALSE \
     -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
     -DTPL_ENABLE_LAPACKLIB=ON \
     -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
     -DTPL_BLAS_LIBRARIES=/opt/cray/pe/libsci/22.11.1.2/GNU/9.1/x86_64/lib/libsci_gnu_82_mp.so \
     -DTPL_LAPACK_LIBRARIES=/opt/cray/pe/libsci/22.11.1.2/GNU/9.1/x86_64/lib/libsci_gnu_82_mp.so \
     -DBUILD_SHARED_LIBS=OFF \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=.
     
make pddrive
make pddrive3d
make f_pddrive


# -DTPL_BLAS_LIBRARIES=/global/cfs/cdirs/m3894/ptlin/tpl/amd_blis/install/amd_blis-20211021-n9-gcc9.3.0/lib/libblis.a \