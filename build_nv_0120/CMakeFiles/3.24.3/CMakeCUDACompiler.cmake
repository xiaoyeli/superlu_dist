set(CMAKE_CUDA_COMPILER "/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/opt/cray/pe/craype/2.7.19/bin/CC")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "11.7.64")
set(CMAKE_CUDA_DEVICE_LINKER "/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "14")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "7.5")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7")
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "11.7.64")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7")

set(CMAKE_CUDA_ARCHITECTURES_ALL "35-real;37-real;50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real;75-real;80-real;86-real;87")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "35-real;50-real;60-real;70-real;80")
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "80-real")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/targets/x86_64-linux/lib/stubs;/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "cupti;cuda;sci_nvidia_mpi;sci_nvidia;mpi_nvidia;mpi_gtl_cuda;dsmml;xpmem;acchost;accdevaux;accdevice;cudadevice;atomic;nvhpcatm;stdc++;nvf;nvomp;nvhpcatm;atomic;nvcpumath;nsnvc;nvc;gcc;c;gcc_s;m")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/targets/x86_64-linux/lib/stubs;/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/targets/x86_64-linux/lib;/opt/cray/pe/mpich/8.1.22/ofi/nvidia/20.7/lib;/opt/cray/pe/mpich/8.1.22/gtl/lib;/opt/cray/pe/libsci/22.11.1.2/NVIDIA/20.7/x86_64/lib;/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/lib64/stubs;/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/lib64;/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/nvvm/lib64;/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/extras/CUPTI/lib64;/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/extras/Debugger/lib64;/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/math_libs/11.7/lib64;/opt/cray/pe/dsmml/0.2.2/dsmml/lib;/opt/cray/xpmem/2.5.2-2.4_3.20__gd0f7936.shasta/lib64;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/compilers/lib;/usr/lib64;/usr/lib64/gcc/x86_64-suse-linux/7")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_MT "")
