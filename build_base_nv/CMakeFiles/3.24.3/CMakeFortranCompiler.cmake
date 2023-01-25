set(CMAKE_Fortran_COMPILER "/opt/cray/pe/craype/2.7.19/bin/ftn")
set(CMAKE_Fortran_COMPILER_ARG1 "")
set(CMAKE_Fortran_COMPILER_ID "NVHPC")
set(CMAKE_Fortran_COMPILER_VERSION "22.7.0")
set(CMAKE_Fortran_COMPILER_WRAPPER "CrayPrgEnv")
set(CMAKE_Fortran_PLATFORM_ID "Linux")
set(CMAKE_Fortran_SIMULATE_ID "")
set(CMAKE_Fortran_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_Fortran_SIMULATE_VERSION "")




set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_Fortran_COMPILER_AR "")
set(CMAKE_RANLIB "/usr/bin/ranlib")
set(CMAKE_Fortran_COMPILER_RANLIB "")
set(CMAKE_COMPILER_IS_GNUG77 )
set(CMAKE_Fortran_COMPILER_LOADED 1)
set(CMAKE_Fortran_COMPILER_WORKS TRUE)
set(CMAKE_Fortran_ABI_COMPILED TRUE)

set(CMAKE_Fortran_COMPILER_ENV_VAR "FC")

set(CMAKE_Fortran_COMPILER_SUPPORTS_F90 1)

set(CMAKE_Fortran_COMPILER_ID_RUN 1)
set(CMAKE_Fortran_SOURCE_FILE_EXTENSIONS f;F;fpp;FPP;f77;F77;f90;F90;for;For;FOR;f95;F95)
set(CMAKE_Fortran_IGNORE_EXTENSIONS h;H;o;O;obj;OBJ;def;DEF;rc;RC)
set(CMAKE_Fortran_LINKER_PREFERENCE 20)
if(UNIX)
  set(CMAKE_Fortran_OUTPUT_EXTENSION .o)
else()
  set(CMAKE_Fortran_OUTPUT_EXTENSION .obj)
endif()

# Save compiler ABI information.
set(CMAKE_Fortran_SIZEOF_DATA_PTR "8")
set(CMAKE_Fortran_COMPILER_ABI "")
set(CMAKE_Fortran_LIBRARY_ARCHITECTURE "")

if(CMAKE_Fortran_SIZEOF_DATA_PTR AND NOT CMAKE_SIZEOF_VOID_P)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_Fortran_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_Fortran_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_Fortran_COMPILER_ABI}")
endif()

if(CMAKE_Fortran_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()





set(CMAKE_Fortran_IMPLICIT_INCLUDE_DIRECTORIES "")
set(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES "cupti;cudart;cuda;sci_nvidia_mpi;sci_nvidia;dl;mpifort_nvidia;mpi_nvidia;mpi_gtl_cuda;dsmml;xpmem;acchost;accdevaux;accdevice;dl;cudadevice;nvf;nvomp;dl;nvhpcatm;atomic;pthread;nvcpumath;nsnvc;nvc;rt;pthread;gcc;c;gcc_s;m")
set(CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES "/opt/cray/pe/mpich/8.1.22/ofi/nvidia/20.7/lib;/opt/cray/pe/mpich/8.1.22/gtl/lib;/opt/cray/pe/libsci/22.11.1.2/NVIDIA/20.7/x86_64/lib;/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/lib64/stubs;/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/lib64;/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/nvvm/lib64;/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/extras/CUPTI/lib64;/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/extras/Debugger/lib64;/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/math_libs/11.7/lib64;/opt/cray/pe/dsmml/0.2.2/dsmml/lib;/opt/cray/xpmem/2.5.2-2.4_3.20__gd0f7936.shasta/lib64;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/compilers/lib;/usr/lib64;/usr/lib64/gcc/x86_64-suse-linux/7")
set(CMAKE_Fortran_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
