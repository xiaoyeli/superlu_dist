set(CMAKE_C_COMPILER "/soft/restricted/CNDA/modulefiles/oneapi/bin/icx")
set(CMAKE_C_COMPILER_ARG1 "")
set(CMAKE_C_COMPILER_ID "Clang")
set(CMAKE_C_COMPILER_VERSION "12.0.0")
set(CMAKE_C_COMPILER_VERSION_INTERNAL "")
set(CMAKE_C_COMPILER_WRAPPER "")
set(CMAKE_C_STANDARD_COMPUTED_DEFAULT "99")
set(CMAKE_C_COMPILE_FEATURES "c_std_90;c_function_prototypes;c_std_99;c_restrict;c_variadic_macros;c_std_11;c_static_assert")
set(CMAKE_C90_COMPILE_FEATURES "c_std_90;c_function_prototypes")
set(CMAKE_C99_COMPILE_FEATURES "c_std_99;c_restrict;c_variadic_macros")
set(CMAKE_C11_COMPILE_FEATURES "c_std_11;c_static_assert")

set(CMAKE_C_PLATFORM_ID "Linux")
set(CMAKE_C_SIMULATE_ID "")
set(CMAKE_C_COMPILER_FRONTEND_VARIANT "GNU")
set(CMAKE_C_SIMULATE_VERSION "")




set(CMAKE_AR "/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/binutils-2.34-rnwhrdgiqluqiypg5palnxdxviv3mynt/bin/ar")
set(CMAKE_C_COMPILER_AR "/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/compiler/20210130/linux/bin/llvm-ar")
set(CMAKE_RANLIB "/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/binutils-2.34-rnwhrdgiqluqiypg5palnxdxviv3mynt/bin/ranlib")
set(CMAKE_C_COMPILER_RANLIB "/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/compiler/20210130/linux/bin/llvm-ranlib")
set(CMAKE_LINKER "/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/binutils-2.34-rnwhrdgiqluqiypg5palnxdxviv3mynt/bin/ld")
set(CMAKE_MT "")
set(CMAKE_COMPILER_IS_GNUCC )
set(CMAKE_C_COMPILER_LOADED 1)
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_C_ABI_COMPILED TRUE)
set(CMAKE_COMPILER_IS_MINGW )
set(CMAKE_COMPILER_IS_CYGWIN )
if(CMAKE_COMPILER_IS_CYGWIN)
  set(CYGWIN 1)
  set(UNIX 1)
endif()

set(CMAKE_C_COMPILER_ENV_VAR "CC")

if(CMAKE_COMPILER_IS_MINGW)
  set(MINGW 1)
endif()
set(CMAKE_C_COMPILER_ID_RUN 1)
set(CMAKE_C_SOURCE_FILE_EXTENSIONS c;m)
set(CMAKE_C_IGNORE_EXTENSIONS h;H;o;O;obj;OBJ;def;DEF;rc;RC)
set(CMAKE_C_LINKER_PREFERENCE 10)

# Save compiler ABI information.
set(CMAKE_C_SIZEOF_DATA_PTR "8")
set(CMAKE_C_COMPILER_ABI "ELF")
set(CMAKE_C_LIBRARY_ARCHITECTURE "")

if(CMAKE_C_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_C_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_C_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_C_COMPILER_ABI}")
endif()

if(CMAKE_C_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "")
if(CMAKE_C_CL_SHOWINCLUDES_PREFIX)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_C_CL_SHOWINCLUDES_PREFIX}")
endif()





set(CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES "/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/dpl/2021.1.1/linux/include;/soft/libraries/hwloc/2.3.0/include;/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/include;/soft/perftools/intel-md/master-Release-2021.02.03/include;/soft/libraries/intel-gmmlib/6f15b79-Release-2021.02.03/include;/soft/libraries/intel-level-zero/api_+_loader/e2b2969-Release-2021.02.03/include/level_zero;/soft/libraries/intel-level-zero/api_+_loader/e2b2969-Release-2021.02.03/include;/soft/libraries/khronos/headers/master-2021.02.03/include;/soft/compilers/intel-igc/79928b2-Release-2021.02.03/include;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/dpcpp-ct/2021.1.1/include;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/dal/2021.1.3/include;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/dnnl/2021.1.1-prerelease/cpu_dpcpp_gpu_dpcpp/include;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/tbb/2021.2.0/include;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/mkl/20210112/include;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/compiler/20210130/linux/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/gcc-9.3.0-qfmcwfbuvnpn47zxjzfjvodzjl6reerh/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/mpfr-3.1.6-pcot5yhbogy4kf5cmjmgc2m4sob33wtz/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/isl-0.20-fey2k52cci6htekjlh3jitmcdvwckvb6/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/binutils-2.34-rnwhrdgiqluqiypg5palnxdxviv3mynt/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/gettext-0.20.1-4uprshc3uyvqflnqxluhosm7rowjzui3/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/libxml2-2.9.9-2yk5s47fz25ngwxye6jx6ukvntdnb4af/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/xz-5.2.5-6rgt4w7in65lwhal6husiesraanx4dou/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/libiconv-1.16-zqp7gscw3ojfabhhzbeco7t26fuwypol/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/bzip2-1.0.8-vatuhzamrqipcsl3cutrbfrw74du4amr/include;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/compiler/20210130/linux/compiler/include;/usr/local/include;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/compiler/20210130/linux/lib/clang/12.0.0/include;/usr/include")
set(CMAKE_C_IMPLICIT_LINK_LIBRARIES "svml;irng;imf;m;gcc;gcc_s;irc;dl;gcc;gcc_s;c;gcc;gcc_s;irc_s")
set(CMAKE_C_IMPLICIT_LINK_DIRECTORIES "/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/compiler/20210130/linux/compiler/lib/intel64_lin;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/compiler/20210130/linux/lib;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/gcc-9.3.0-qfmcwfbuvnpn47zxjzfjvodzjl6reerh/lib/gcc/x86_64-pc-linux-gnu/9.3.0;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/gcc-9.3.0-qfmcwfbuvnpn47zxjzfjvodzjl6reerh/lib64;/lib64;/usr/lib64;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/gcc-9.3.0-qfmcwfbuvnpn47zxjzfjvodzjl6reerh/lib;/lib;/usr/lib;/soft/libraries/hwloc/2.3.0/lib;/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/lib;/soft/perftools/intel-md/master-Release-2021.02.03/lib64;/soft/libraries/khronos/loader/master-2021.02.03/lib64;/soft/libraries/intel-level-zero/api_+_loader/e2b2969-Release-2021.02.03/lib64;/soft/libraries/intel-gmmlib/6f15b79-Release-2021.02.03/lib64;/soft/libraries/intel-level-zero/compute-runtime/7a91ef8-Release-2021.02.03/lib64;/soft/libraries/intel-level-zero/compute-runtime/7a91ef8-Release-2021.02.03/lib64/intel-opencl;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/mpfr-3.1.6-pcot5yhbogy4kf5cmjmgc2m4sob33wtz/lib;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/isl-0.20-fey2k52cci6htekjlh3jitmcdvwckvb6/lib;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/binutils-2.34-rnwhrdgiqluqiypg5palnxdxviv3mynt/lib;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/gettext-0.20.1-4uprshc3uyvqflnqxluhosm7rowjzui3/lib;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/libxml2-2.9.9-2yk5s47fz25ngwxye6jx6ukvntdnb4af/lib;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/xz-5.2.5-6rgt4w7in65lwhal6husiesraanx4dou/lib;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/libiconv-1.16-zqp7gscw3ojfabhhzbeco7t26fuwypol/lib;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/bzip2-1.0.8-vatuhzamrqipcsl3cutrbfrw74du4amr/lib;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/dal/2021.1.3/lib/intel64;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/dnnl/2021.1.1-prerelease/cpu_dpcpp_gpu_dpcpp/lib;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/tbb/2021.2.0/lib/intel64/gcc4.8;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/mkl/20210112/lib/intel64")
set(CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
