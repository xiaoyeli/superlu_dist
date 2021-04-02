set(CMAKE_CXX_COMPILER "/soft/restricted/CNDA/modulefiles/oneapi/bin/dpcpp")
set(CMAKE_CXX_COMPILER_ARG1 "")
set(CMAKE_CXX_COMPILER_ID "Clang")
set(CMAKE_CXX_COMPILER_VERSION "12.0.0")
set(CMAKE_CXX_COMPILER_VERSION_INTERNAL "")
set(CMAKE_CXX_COMPILER_WRAPPER "")
set(CMAKE_CXX_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CXX_COMPILE_FEATURES "cxx_std_98;cxx_template_template_parameters;cxx_std_11;cxx_alias_templates;cxx_alignas;cxx_alignof;cxx_attributes;cxx_auto_type;cxx_constexpr;cxx_decltype;cxx_decltype_incomplete_return_types;cxx_default_function_template_args;cxx_defaulted_functions;cxx_defaulted_move_initializers;cxx_delegating_constructors;cxx_deleted_functions;cxx_enum_forward_declarations;cxx_explicit_conversions;cxx_extended_friend_declarations;cxx_extern_templates;cxx_final;cxx_func_identifier;cxx_generalized_initializers;cxx_inheriting_constructors;cxx_inline_namespaces;cxx_lambdas;cxx_local_type_template_args;cxx_long_long_type;cxx_noexcept;cxx_nonstatic_member_init;cxx_nullptr;cxx_override;cxx_range_for;cxx_raw_string_literals;cxx_reference_qualified_functions;cxx_right_angle_brackets;cxx_rvalue_references;cxx_sizeof_member;cxx_static_assert;cxx_strong_enums;cxx_thread_local;cxx_trailing_return_types;cxx_unicode_literals;cxx_uniform_initialization;cxx_unrestricted_unions;cxx_user_literals;cxx_variadic_macros;cxx_variadic_templates;cxx_std_14;cxx_aggregate_default_initializers;cxx_attribute_deprecated;cxx_binary_literals;cxx_contextual_conversions;cxx_decltype_auto;cxx_digit_separators;cxx_generic_lambdas;cxx_lambda_init_captures;cxx_relaxed_constexpr;cxx_return_type_deduction;cxx_variable_templates;cxx_std_17;cxx_std_20")
set(CMAKE_CXX98_COMPILE_FEATURES "cxx_std_98;cxx_template_template_parameters")
set(CMAKE_CXX11_COMPILE_FEATURES "cxx_std_11;cxx_alias_templates;cxx_alignas;cxx_alignof;cxx_attributes;cxx_auto_type;cxx_constexpr;cxx_decltype;cxx_decltype_incomplete_return_types;cxx_default_function_template_args;cxx_defaulted_functions;cxx_defaulted_move_initializers;cxx_delegating_constructors;cxx_deleted_functions;cxx_enum_forward_declarations;cxx_explicit_conversions;cxx_extended_friend_declarations;cxx_extern_templates;cxx_final;cxx_func_identifier;cxx_generalized_initializers;cxx_inheriting_constructors;cxx_inline_namespaces;cxx_lambdas;cxx_local_type_template_args;cxx_long_long_type;cxx_noexcept;cxx_nonstatic_member_init;cxx_nullptr;cxx_override;cxx_range_for;cxx_raw_string_literals;cxx_reference_qualified_functions;cxx_right_angle_brackets;cxx_rvalue_references;cxx_sizeof_member;cxx_static_assert;cxx_strong_enums;cxx_thread_local;cxx_trailing_return_types;cxx_unicode_literals;cxx_uniform_initialization;cxx_unrestricted_unions;cxx_user_literals;cxx_variadic_macros;cxx_variadic_templates")
set(CMAKE_CXX14_COMPILE_FEATURES "cxx_std_14;cxx_aggregate_default_initializers;cxx_attribute_deprecated;cxx_binary_literals;cxx_contextual_conversions;cxx_decltype_auto;cxx_digit_separators;cxx_generic_lambdas;cxx_lambda_init_captures;cxx_relaxed_constexpr;cxx_return_type_deduction;cxx_variable_templates")
set(CMAKE_CXX17_COMPILE_FEATURES "cxx_std_17")
set(CMAKE_CXX20_COMPILE_FEATURES "cxx_std_20")

set(CMAKE_CXX_PLATFORM_ID "Linux")
set(CMAKE_CXX_SIMULATE_ID "")
set(CMAKE_CXX_COMPILER_FRONTEND_VARIANT "GNU")
set(CMAKE_CXX_SIMULATE_VERSION "")




set(CMAKE_AR "/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/binutils-2.34-rnwhrdgiqluqiypg5palnxdxviv3mynt/bin/ar")
set(CMAKE_CXX_COMPILER_AR "/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/compiler/20210130/linux/bin/llvm-ar")
set(CMAKE_RANLIB "/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/binutils-2.34-rnwhrdgiqluqiypg5palnxdxviv3mynt/bin/ranlib")
set(CMAKE_CXX_COMPILER_RANLIB "/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/compiler/20210130/linux/bin/llvm-ranlib")
set(CMAKE_LINKER "/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/binutils-2.34-rnwhrdgiqluqiypg5palnxdxviv3mynt/bin/ld")
set(CMAKE_MT "")
set(CMAKE_COMPILER_IS_GNUCXX )
set(CMAKE_CXX_COMPILER_LOADED 1)
set(CMAKE_CXX_COMPILER_WORKS TRUE)
set(CMAKE_CXX_ABI_COMPILED TRUE)
set(CMAKE_COMPILER_IS_MINGW )
set(CMAKE_COMPILER_IS_CYGWIN )
if(CMAKE_COMPILER_IS_CYGWIN)
  set(CYGWIN 1)
  set(UNIX 1)
endif()

set(CMAKE_CXX_COMPILER_ENV_VAR "CXX")

if(CMAKE_COMPILER_IS_MINGW)
  set(MINGW 1)
endif()
set(CMAKE_CXX_COMPILER_ID_RUN 1)
set(CMAKE_CXX_SOURCE_FILE_EXTENSIONS C;M;c++;cc;cpp;cxx;m;mm;CPP)
set(CMAKE_CXX_IGNORE_EXTENSIONS inl;h;hpp;HPP;H;o;O;obj;OBJ;def;DEF;rc;RC)

foreach (lang C OBJC OBJCXX)
  if (CMAKE_${lang}_COMPILER_ID_RUN)
    foreach(extension IN LISTS CMAKE_${lang}_SOURCE_FILE_EXTENSIONS)
      list(REMOVE_ITEM CMAKE_CXX_SOURCE_FILE_EXTENSIONS ${extension})
    endforeach()
  endif()
endforeach()

set(CMAKE_CXX_LINKER_PREFERENCE 30)
set(CMAKE_CXX_LINKER_PREFERENCE_PROPAGATES 1)

# Save compiler ABI information.
set(CMAKE_CXX_SIZEOF_DATA_PTR "8")
set(CMAKE_CXX_COMPILER_ABI "ELF")
set(CMAKE_CXX_LIBRARY_ARCHITECTURE "")

if(CMAKE_CXX_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CXX_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CXX_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CXX_COMPILER_ABI}")
endif()

if(CMAKE_CXX_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "")
if(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_CXX_CL_SHOWINCLUDES_PREFIX}")
endif()





set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/dpl/2021.1.1/linux/include;/soft/libraries/hwloc/2.3.0/include;/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/include;/soft/perftools/intel-md/master-Release-2021.02.03/include;/soft/libraries/intel-gmmlib/6f15b79-Release-2021.02.03/include;/soft/libraries/intel-level-zero/api_+_loader/e2b2969-Release-2021.02.03/include/level_zero;/soft/libraries/intel-level-zero/api_+_loader/e2b2969-Release-2021.02.03/include;/soft/libraries/khronos/headers/master-2021.02.03/include;/soft/compilers/intel-igc/79928b2-Release-2021.02.03/include;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/dpcpp-ct/2021.1.1/include;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/dal/2021.1.3/include;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/dnnl/2021.1.1-prerelease/cpu_dpcpp_gpu_dpcpp/include;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/tbb/2021.2.0/include;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/mkl/20210112/include;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/compiler/20210130/linux/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/gcc-9.3.0-qfmcwfbuvnpn47zxjzfjvodzjl6reerh/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/mpfr-3.1.6-pcot5yhbogy4kf5cmjmgc2m4sob33wtz/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/isl-0.20-fey2k52cci6htekjlh3jitmcdvwckvb6/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/binutils-2.34-rnwhrdgiqluqiypg5palnxdxviv3mynt/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/gettext-0.20.1-4uprshc3uyvqflnqxluhosm7rowjzui3/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/libxml2-2.9.9-2yk5s47fz25ngwxye6jx6ukvntdnb4af/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/xz-5.2.5-6rgt4w7in65lwhal6husiesraanx4dou/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/libiconv-1.16-zqp7gscw3ojfabhhzbeco7t26fuwypol/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/bzip2-1.0.8-vatuhzamrqipcsl3cutrbfrw74du4amr/include;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/compiler/20210130/linux/include/sycl;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/compiler/20210130/linux/compiler/include;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/gcc-9.3.0-qfmcwfbuvnpn47zxjzfjvodzjl6reerh/include/c++/9.3.0;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/gcc-9.3.0-qfmcwfbuvnpn47zxjzfjvodzjl6reerh/include/c++/9.3.0/x86_64-pc-linux-gnu;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/gcc-9.3.0-qfmcwfbuvnpn47zxjzfjvodzjl6reerh/include/c++/9.3.0/backward;/usr/local/include;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/compiler/20210130/linux/lib/clang/12.0.0/include;/usr/include")
set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES "svml;irng;stdc++;imf;m;gcc_s;gcc;irc;dl;gcc_s;gcc;sycl;c;gcc_s;gcc;irc_s")
set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/compiler/20210130/linux/compiler/lib/intel64_lin;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/compiler/20210130/linux/lib;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/gcc-9.3.0-qfmcwfbuvnpn47zxjzfjvodzjl6reerh/lib/gcc/x86_64-pc-linux-gnu/9.3.0;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/gcc-9.3.0-qfmcwfbuvnpn47zxjzfjvodzjl6reerh/lib64;/lib64;/usr/lib64;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/gcc-9.3.0-qfmcwfbuvnpn47zxjzfjvodzjl6reerh/lib;/lib;/usr/lib;/soft/libraries/hwloc/2.3.0/lib;/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/lib;/soft/perftools/intel-md/master-Release-2021.02.03/lib64;/soft/libraries/khronos/loader/master-2021.02.03/lib64;/soft/libraries/intel-level-zero/api_+_loader/e2b2969-Release-2021.02.03/lib64;/soft/libraries/intel-gmmlib/6f15b79-Release-2021.02.03/lib64;/soft/libraries/intel-level-zero/compute-runtime/7a91ef8-Release-2021.02.03/lib64;/soft/libraries/intel-level-zero/compute-runtime/7a91ef8-Release-2021.02.03/lib64/intel-opencl;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/mpfr-3.1.6-pcot5yhbogy4kf5cmjmgc2m4sob33wtz/lib;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/isl-0.20-fey2k52cci6htekjlh3jitmcdvwckvb6/lib;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/binutils-2.34-rnwhrdgiqluqiypg5palnxdxviv3mynt/lib;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/gettext-0.20.1-4uprshc3uyvqflnqxluhosm7rowjzui3/lib;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/libxml2-2.9.9-2yk5s47fz25ngwxye6jx6ukvntdnb4af/lib;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/xz-5.2.5-6rgt4w7in65lwhal6husiesraanx4dou/lib;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/libiconv-1.16-zqp7gscw3ojfabhhzbeco7t26fuwypol/lib;/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/bzip2-1.0.8-vatuhzamrqipcsl3cutrbfrw74du4amr/lib;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/dal/2021.1.3/lib/intel64;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/dnnl/2021.1.1-prerelease/cpu_dpcpp_gpu_dpcpp/lib;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/tbb/2021.2.0/lib/intel64/gcc4.8;/soft/restricted/CNDA/sdk/2020.12.15.1-ats/oneapi/mkl/20210112/lib/intel64")
set(CMAKE_CXX_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
