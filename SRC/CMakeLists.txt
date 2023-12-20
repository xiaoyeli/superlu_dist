set(headers
    include/superlu_FCnames.h
    include/dcomplex.h
    include/machines.h
    include/psymbfact.h
    include/superlu_defs.h
    include/superlu_enum_consts.h
    include/supermatrix.h
    include/util_dist.h
    include/gpu_api_utils.h
    include/gpu_wrapper.h
    include/superlu_upacked.h
    include/superlu_dist_config.h
    include/superlu_FortranCInterface.h
    TRF3dV100/lupanels.hpp
    TRF3dV100/commWrapper.hpp
    TRF3dV100/lupanels_GPU.cuh
    TRF3dV100/batch_block_copy.h
    TRF3dV100/anc25d-GPU_impl.hpp
    TRF3dV100/anc25d.hpp
    TRF3dV100/lupanelsComm3dGPU_impl.hpp
    TRF3dV100/anc25d_impl.hpp
    TRF3dV100/lupanels_GPU_impl.hpp
    TRF3dV100/lupanels_comm3d_impl.hpp
    TRF3dV100/cublas_cusolver_wrappers.hpp
    TRF3dV100/lupanels_impl.hpp
    TRF3dV100/dAncestorFactor_impl.hpp
    TRF3dV100/pdgstrf3d_upacked_impl.hpp
    TRF3dV100/dsparseTreeFactorGPU_impl.hpp
    TRF3dV100/sparseTreeFactor_impl.hpp
    TRF3dV100/dsparseTreeFactor_upacked_impl.hpp
    TRF3dV100/superlu_blas.hpp
    TRF3dV100/l_panels_impl.hpp
    TRF3dV100/u_panels_impl.hpp
    TRF3dV100/luAuxStructTemplated.hpp
    TRF3dV100/xlupanels.hpp
    TRF3dV100/lu_common.hpp
    TRF3dV100/xlupanels_GPU.cuh
    TRF3dV100/schurCompUpdate_impl.cuh 
)
#    ${CMAKE_CURRENT_BINARY_DIR}/superlu_dist_config.h
#    ${PROJECT_SOURCE_DIR}/SRC/superlu_FortranCInterface.h
#    colamd.h
if (MSVC)
  list(APPEND headers wingetopt.h)
endif ()

# first: precision-independent files
set(sources
  prec-independent/sp_ienv.c
  prec-independent/etree.c 
  prec-independent/sp_colorder.c
  prec-independent/get_perm_c.c
  prec-independent/mmd.c
  prec-independent/comm.c
  prec-independent/memory.c
  prec-independent/util.c
  prec-independent/gpu_api_utils.c
  prec-independent/superlu_grid.c
  prec-independent/pxerr_dist.c
  prec-independent/superlu_timer.c
  prec-independent/symbfact.c
  prec-independent/psymbfact.c
  prec-independent/psymbfact_util.c
  prec-independent/get_perm_c_parmetis.c
  prec-independent/mc64ad_dist.c
  prec-independent/xerr_dist.c
  prec-independent/smach_dist.c
  prec-independent/dmach_dist.c
  prec-independent/superlu_dist_version.c
  prec-independent/comm_tree.c
  prec-independent/superlu_grid3d.c    ## 3D code
  prec-independent/supernodal_etree.c
  prec-independent/supernodalForest.c
  prec-independent/trfAux.c 
  prec-independent/communication_aux.c
  prec-independent/treeFactorization.c
  prec-independent/sec_structs.c  
)

# if (HAVE_CUDA)
#   list(APPEND sources cuda/superlu_gpu_utils.cu)
# endif()

# if (HAVE_HIP)
#   list(APPEND sources hip/superlu_gpu_utils.hip.cpp)
# endif()

if (MSVC)
  list(APPEND sources pred-independent/wingetopt.c)
endif ()

set_source_files_properties(superlu_timer.c PROPERTIES COMPILE_FLAGS -O0)

if(enable_double)
  list(APPEND headers
  include/superlu_ddefs.h include/dlustruct_gpu.h include/oneside.h)

if(TPL_ENABLE_CUDALIB)
  list(APPEND sources cuda/pdgstrs_lsum_cuda.cu)
  list(APPEND sources TRF3dV100/schurCompUpdate.cu )
  list(APPEND sources TRF3dV100/batch_block_copy.cu)
endif()
# if(TPL_ENABLE_HIPLIB)
#   list(APPEND sources hip/pdgstrs_lsum_cuda.hip.cpp)
# endif()    	

  list(APPEND sources
    double/dlangs_dist.c
    double/dgsequ_dist.c
    double/dlaqgs_dist.c
    double/dutil_dist.c
    double/dmemory_dist.c
    double/dmyblas2_dist.c
    double/dsp_blas2_dist.c
    double/dsp_blas3_dist.c
    double/pdgssvx.c
    double/pdgssvx_ABglobal.c
    double/dreadhb.c
    double/dreadrb.c
    double/dreadtriple.c
    double/dreadtriple_noheader.c
    double/dbinary_io.c	
    double/dreadMM.c
    double/pdgsequ.c
    double/pdlaqgs.c
    double/dldperm_dist.c
    double/pdlangs.c
    double/pdutil.c
    double/pdsymbfact_distdata.c
    double/ddistribute.c
    double/pddistribute.c
    double/pddistribute3d.c
    double/d3DPartition.c
    double/distCheckArray.c
    double/pddistribute-aux3d.c
    double/pdgstrf.c
    double/dstatic_schedule.c
    double/pdgstrf2.c
    double/pdgstrs.c
    double/pdgstrs3d.c
    double/pdgstrs1.c
    double/pdgstrs_lsum.c
    double/pdgstrs_Bglobal.c
    double/pdgsrfs.c
    double/pdgsmv.c
    double/pdgsrfs_ABXglobal.c
    double/pdgsmv_AXglobal.c
    double/pdGetDiagU.c
    double/pdgssvx3d.c     ## 3D code
    double/dnrformat_loc3d.c 
    double/pdgstrf3d.c 
    double/dtreeFactorization.c
    double/dtreeFactorizationGPU.c
    double/dgather.c
    double/dscatter3d.c
    double/pd3dcomm.c
    double/dtrfAux.c	
    double/dcommunication_aux.c 
    double/dtrfCommWrapper.c
    double/dsuperlu_blas.c
  )
if (TPL_ENABLE_CUDALIB)
  list(APPEND sources
       cuda/pdgstrs_lsum_cuda.cu cuda/superlu_gpu_utils.cu cuda/dsuperlu_gpu.cu
  TRF3dV100/schurCompUpdate.cu 
  TRF3dV100/pdgstrf3d_summit.cpp
    TRF3dV100/dsparseTreeFactor_upacked.cpp    TRF3dV100/dsparseTreeFactorGPU.cpp
    TRF3dV100/anc25d-GPU.cpp 
    TRF3dV100/lupanels.cpp
    TRF3dV100/lupanels_comm3d.cpp
    TRF3dV100/lupanelsComm3dGPU.cpp
    TRF3dV100/commWrapper.cpp 
    TRF3dV100/l_panels.cpp
    TRF3dV100/u_panels.cpp
    TRF3dV100/lupanels_GPU.cpp
    TRF3dV100/anc25d.cpp 
    TRF3dV100/LUgpuCHandle_interface.cpp
    TRF3dV100/LUgpuCHandle_interface_impl.cu  
  )
endif()

if (HAVE_COMBBLAS)
  list(APPEND sources double/d_c2cpp_GetHWPM.cpp double/dHWPM_CombBLAS.hpp)
endif()

endif() ########## enable double

if(enable_single)
  list(APPEND headers
       include/superlu_sdefs.h include/slustruct_gpu.h)

  list(APPEND sources
    single/slangs_dist.c
    single/sgsequ_dist.c
    single/slaqgs_dist.c
    single/sutil_dist.c
    single/smemory_dist.c
    single/smyblas2_dist.c
    single/ssp_blas2_dist.c
    single/ssp_blas3_dist.c
    single/psgssvx.c
    single/psgssvx_d2.c
    single/psgssvx_ABglobal.c
    single/sreadhb.c
    single/sreadrb.c
    single/sreadtriple.c
    single/sreadtriple_noheader.c
    single/sbinary_io.c	
    single/sreadMM.c
    single/psgsequ.c
    single/pslaqgs.c
    single/sldperm_dist.c
    single/pslangs.c
    single/psutil.c
    single/pssymbfact_distdata.c
    single/sdistribute.c
    single/psdistribute.c
    single/psgstrf.c
    single/sstatic_schedule.c
    single/psgstrf2.c
    single/psgstrs.c
    single/psgstrs1.c
    single/psgstrs_lsum.c
    single/psgstrs_Bglobal.c
    single/psgsrfs.c
    single/psgsrfs_d2.c
    single/psgsmv.c
    single/psgsrfs_ABXglobal.c
    single/psgsmv_AXglobal.c
    single/psGetDiagU.c
    single/psgssvx3d.c     ## 3D code
    single/snrformat_loc3d.c 
    single/psgstrf3d.c 
    single/streeFactorization.c
    single/streeFactorizationGPU.c
    single/sgather.c
    single/sscatter3d.c
    single/ps3dcomm.c
    single/strfAux.c	
    single/scommunication_aux.c 
    single/strfCommWrapper.c
    single/ssuperlu_blas.c
    single/psgssvx_d2.c     # with double-precision IR
    single/psgsrfs_d2.c
    single/psgsmv_d2.c
    single/psgsequb.c
  )
if (TPL_ENABLE_CUDALIB)
  list(APPEND sources cuda/ssuperlu_gpu.cu)
endif()
if (HAVE_COMBBLAS)
   if (enable_double)
       list(APPEND sources double/d_c2cpp_GetHWPM.cpp)
   else()
       list(APPEND sources single/s_c2cpp_GetHWPM.cpp double/dHWPM_CombBLAS.hpp)
   endif()
endif()

endif() ########## enable single


if(enable_complex16)
  list(APPEND headers
       include/superlu_zdefs.h include/zlustruct_gpu.h)

  list(APPEND sources
    complex16/dcomplex_dist.c
    complex16/zlangs_dist.c
    complex16/zgsequ_dist.c
    complex16/zlaqgs_dist.c
    complex16/zutil_dist.c
    complex16/zmemory_dist.c
    complex16/zmyblas2_dist.c
    complex16/zsp_blas2_dist.c
    complex16/zsp_blas3_dist.c
    complex16/pzgssvx.c
    complex16/pzgssvx_ABglobal.c
    complex16/zreadhb.c
    complex16/zreadrb.c
    complex16/zreadtriple.c
    complex16/zreadtriple_noheader.c
    complex16/zbinary_io.c	
    complex16/zreadMM.c
    complex16/pzgsequ.c
    complex16/pzlaqgs.c
    complex16/zldperm_dist.c
    complex16/pzlangs.c
    complex16/pzutil.c
    complex16/pzsymbfact_distdata.c
    complex16/zdistribute.c
    complex16/pzdistribute.c
    complex16/pzgstrf.c
    complex16/zstatic_schedule.c
    complex16/pzgstrf2.c
    complex16/pzgstrs.c
    complex16/pzgstrs1.c
    complex16/pzgstrs_lsum.c
    complex16/pzgstrs_Bglobal.c
    complex16/pzgsrfs.c
    complex16/pzgsmv.c
    complex16/pzgsrfs_ABXglobal.c
    complex16/pzgsmv_AXglobal.c
    complex16/pzGetDiagU.c
    complex16/pzgssvx3d.c   ## 3D code
    complex16/znrformat_loc3d.c 
    complex16/pzgstrf3d.c
    complex16/ztreeFactorization.c 
    complex16/ztreeFactorizationGPU.c 
    complex16/zscatter3d.c
    complex16/zgather.c 
    complex16/pz3dcomm.c
    complex16/ztrfAux.c
    complex16/zcommunication_aux.c
    complex16/ztrfCommWrapper.c
    complex16/zsuperlu_blas.c
  )
if (TPL_ENABLE_CUDALIB)
    list(APPEND sources cuda/zsuperlu_gpu.cu)
endif()
if (HAVE_COMBBLAS)
    list(APPEND sources
         complex16/z_c2cpp_GetHWPM.cpp complex16/zHWPM_CombBLAS.hpp)
endif()
endif() ######### enable compex16

if (TPL_ENABLE_HIPLIB)
  file(GLOB MyFiles hip/*.hip.cpp)
  set_source_files_properties(
    ${MyFiles}
    PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

  set(hipsources
    hip/pdgstrs_lsum_cuda.hip.cpp
    hip/dsuperlu_gpu.hip.cpp
    hip/superlu_gpu_utils.hip.cpp
  )
  if(enable_single)
  list(APPEND hipsources hip/ssuperlu_gpu.hip.cpp)
  endif()
  if(enable_complex16)
  list(APPEND hipsources hip/zsuperlu_gpu.hip.cpp)
  endif()
  hip_add_library(superlu_dist ${hipsources})
  if (BUILD_SHARED_LIBS AND BUILD_STATIC_LIBS)
    hip_add_library(superlu_dist-static STATIC  ${hipsources})
  endif()

else()
  add_library(superlu_dist "")
  if (BUILD_SHARED_LIBS AND BUILD_STATIC_LIBS)
    add_library(superlu_dist-static STATIC "")
  endif()
endif()


target_sources(superlu_dist PRIVATE ${sources} ${HEADERS})
set(targets superlu_dist)

if (BUILD_SHARED_LIBS AND BUILD_STATIC_LIBS)
  # build both shared and static libs
  target_sources(superlu_dist-static PRIVATE ${sources} ${HEADERS})
  list(APPEND targets superlu_dist-static)
endif()

if (TPL_ENABLE_NVSHMEM)
set(superlu_dist_libs ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${BLAS_LIB} ${LAPACK_LIB}
    ${PARMETIS_LIB} ${COLAMD_LIB} ${COMBBLAS_LIB} ${MAGMA_LIB} ${CUDA_LIB} ${HIP_LIB} ${NVSHMEM_LIB}) 
else()
set(superlu_dist_libs ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${BLAS_LIB} ${LAPACK_LIB}
    ${PARMETIS_LIB} ${COLAMD_LIB} ${COMBBLAS_LIB} ${MAGMA_LIB} ${CUDA_LIB} ${HIP_LIB})
endif()

if (NOT MSVC)
  list(APPEND superlu_dist_libs m)
endif ()

foreach(target ${targets})
    target_link_libraries(${target} ${superlu_dist_libs})
    if (HAVE_COMBBLAS)
        set_target_properties(${target} PROPERTIES
                              CUDA_SEPARABLE_COMPILATION ON
							  OUTPUT_NAME superlu_dist
                              VERSION ${PROJECT_VERSION}
                              SOVERSION ${VERSION_MAJOR}
			      LINKER_LANGUAGE CXX
			      )
    else()
        set_target_properties(${target} PROPERTIES
		                      CUDA_SEPARABLE_COMPILATION ON
                              OUTPUT_NAME superlu_dist
                              VERSION ${PROJECT_VERSION}
                              SOVERSION ${VERSION_MAJOR}
			      )
    endif()
endforeach(target)

# Add CUDA runtime library and CUBLAS library
if(CUDAToolkit_FOUND)  # this is found in top-level CMakeLists.txt
    target_link_libraries(superlu_dist CUDA::cudart CUDA::cublas)
endif()

# This is recommended by modern cmake:
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
if(OpenMP_FOUND) # this is found in top-level CMakeLists.txt
  target_link_libraries(superlu_dist OpenMP::OpenMP_C)
endif()

if (XSDK_ENABLE_Fortran)
## target_link_libraries(superlu_dist PUBLIC MPI::MPI_CXX MPI::MPI_C MPI::MPI_Fortran)
## PUBLIC keyword causes error:
##   The plain signature for target_link_libraries has already been used ...
  target_link_libraries(superlu_dist MPI::MPI_CXX MPI::MPI_C MPI::MPI_Fortran)
else()
  target_link_libraries(superlu_dist MPI::MPI_CXX MPI::MPI_C)
endif()

target_compile_definitions(superlu_dist PRIVATE SUPERLU_DIST_EXPORTS)
if(MSVC AND BUILD_SHARED_LIBS)
  set_target_properties(superlu_dist PROPERTIES
                        WINDOWS_EXPORT_ALL_SYMBOLS ON
  )
endif()

# Define GNU standard installation directories
include(GNUInstallDirs)

install(TARGETS ${targets}
# DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
)
install(FILES ${headers}
# DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
