 
 
export LIBRARY_PATH=/home/administrator/Desktop/software/llvm-openmp-5/lib/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/home/administrator/Desktop/software/llvm-openmp-5/lib/lib:$LD_LIBRARY_PATH 
 
export PARMETIS_ROOT=/home/administrator/Desktop/software/parmetis-4.0.3-gnu-lomp
export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-x86_64 
rm -rf CMakeCache.txt
rm -rf CMakeFiles
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf DartConfiguration.tcl 
rm -rf SRC
rm -rf EXAMPLE
rm -rf TEST
rm -rf Testing
cmake .. \
	-DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
	-DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.so;${PARMETIS_BUILD_DIR}/libmetis/libmetis.so" \
	-DCMAKE_C_FLAGS="-g -std=c11 -DPRNTlevel=1 -DPROFlevel=1 -DDEBUGlevel=0" \
	-DTPL_BLAS_LIBRARIES="/usr/lib/x86_64-linux-gnu/libblas.so;/usr/lib/x86_64-linux-gnu/liblapack.so" \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_C_COMPILER=mpicc \
	-DCMAKE_INSTALL_PREFIX=. \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DCMAKE_CXX_COMPILER=mpicxx \
	-DCMAKE_CXX_FLAGS="-g -Ofast -std=c++11 -DAdd_ -DRELEASE ${INC_VTUNE}" 
	
# DCMAKE_BUILD_TYPE=Release or Debug compiler options set in CMAKELIST.txt
