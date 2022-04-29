set(headers
    superlu_FCnames.h
    dcomplex.h
    machines.h
    psymbfact.h
    superlu_defs.h
    superlu_enum_consts.h
    supermatrix.h
    util_dist.h
    colamd.h
    gpu_api_utils.h
    gpu_wrapper.h
    ${CMAKE_CURRENT_BINARY_DIR}/superlu_dist_config.h
    ${PROJECT_SOURCE_DIR}/SRC/superlu_FortranCInterface.h
)
if (MSVC)
  list(APPEND headers wingetopt.h)
endif ()

# first: precision-independent files
#  global.cpp
set(sources
  sp_ienv.c
  etree.c 
  sp_colorder.c
  get_perm_c.c
  mmd.c
  comm.c
  memory.c
  util.c
  gpu_api_utils.c
  superlu_grid.c
  pxerr_dist.c
  superlu_timer.c
  symbfact.c
  psymbfact.c
  psymbfact_util.c
  get_perm_c_parmetis.c
  mc64ad_dist.c
  xerr_dist.c
  smach_dist.c
  dmach_dist.c
  colamd.c
  superlu_dist_version.c
  comm_tree.c
  superlu_grid3d.c    ## 3D code
  supernodal_etree.c
  supernodalForest.c
  trfAux.c 
  communication_aux.c
  treeFactorization.c
  sec_structs.c  
)


if (MSVC)
  list(APPEND sources wingetopt.c)
endif ()

set_source_files_properties(superlu_timer.c PROPERTIES COMPILE_FLAGS -O0)

if(enable_double)
  list(APPEND headers superlu_ddefs.h dlustruct_gpu.h)

  list(APPEND sources
    dlangs_dist.c
    dgsequ_dist.c
    dlaqgs_dist.c
    dutil_dist.c
    dmemory_dist.c
    dmyblas2_dist.c
    dsp_blas2_dist.c
    dsp_blas3_dist.c
    pdgssvx.c
    pdgssvx_ABglobal.c
    dreadhb.c
    dreadrb.c
    dreadtriple.c
    dreadtriple_noheader.c
    dbinary_io.c	
    dreadMM.c
    pdgsequ.c
    pdlaqgs.c
    dldperm_dist.c
    pdlangs.c
    pdutil.c
    pdsymbfact_distdata.c
    ddistribute.c
    pddistribute.c
    pdgstrf.c
    dstatic_schedule.c
    pdgstrf2.c
    pdgstrs.c
    pdgstrs1.c
    pdgstrs_lsum.c
    pdgstrs_Bglobal.c
    pdgsrfs.c
    pdgsmv.c
    pdgsrfs_ABXglobal.c
    pdgsmv_AXglobal.c
    pdGetDiagU.c
    pdgssvx3d.c     ## 3D code
    dnrformat_loc3d.c 
    pdgstrf3d.c 
    dtreeFactorization.c
    dtreeFactorizationGPU.c
    dgather.c
    dscatter3d.c
    pd3dcomm.c
    dtrfAux.c	
    dcommunication_aux.c 
    dtrfCommWrapper.c
    dsuperlu_blas.c
  )
if (TPL_ENABLE_CUDALIB)
  list(APPEND sources pdgstrs_lsum_cuda.cu superlu_gpu_utils.cu dsuperlu_gpu.cu)
endif()

if (HAVE_COMBBLAS)
  list(APPEND sources d_c2cpp_GetHWPM.cpp dHWPM_CombBLAS.hpp)
endif()

endif() ########## enable double

if(enable_single)
  list(APPEND headers superlu_sdefs.h slustruct_gpu.h)

  list(APPEND sources
    slangs_dist.c
    sgsequ_dist.c
    slaqgs_dist.c
    sutil_dist.c
    smemory_dist.c
    smyblas2_dist.c
    ssp_blas2_dist.c
    ssp_blas3_dist.c
    psgssvx.c
    psgssvx_d2.c
    psgssvx_ABglobal.c
    sreadhb.c
    sreadrb.c
    sreadtriple.c
    sreadtriple_noheader.c
    sbinary_io.c	
    sreadMM.c
    psgsequ.c
    pslaqgs.c
    sldperm_dist.c
    pslangs.c
    psutil.c
    pssymbfact_distdata.c
    sdistribute.c
    psdistribute.c
    psgstrf.c
    sstatic_schedule.c
    psgstrf2.c
    psgstrs.c
    psgstrs1.c
    psgstrs_lsum.c
    psgstrs_Bglobal.c
    psgsrfs.c
    psgsrfs_d2.c
    psgsmv.c
    psgsrfs_ABXglobal.c
    psgsmv_AXglobal.c
    psGetDiagU.c
    psgssvx3d.c     ## 3D code
    snrformat_loc3d.c 
    psgstrf3d.c 
    streeFactorization.c
    streeFactorizationGPU.c
    sgather.c
    sscatter3d.c
    ps3dcomm.c
    strfAux.c	
    scommunication_aux.c 
    strfCommWrapper.c
    ssuperlu_blas.c
    psgssvx_d2.c     # with double-precision IR
    psgsrfs_d2.c
    psgsmv_d2.c
    psgsequb.c
  )
if (TPL_ENABLE_CUDALIB)
  list(APPEND sources ssuperlu_gpu.cu)
endif()
if (HAVE_COMBBLAS)
  list(APPEND sources s_c2cpp_GetHWPM.cpp sHWPM_CombBLAS.hpp)
endif()

endif() ########## enable single


if(enable_complex16)
  list(APPEND headers superlu_zdefs.h zlustruct_gpu.h)

  list(APPEND sources
    dcomplex_dist.c
    zlangs_dist.c
    zgsequ_dist.c
    zlaqgs_dist.c
    zutil_dist.c
    zmemory_dist.c
    zmyblas2_dist.c
    zsp_blas2_dist.c
    zsp_blas3_dist.c
    pzgssvx.c
    pzgssvx_ABglobal.c
    zreadhb.c
    zreadrb.c
    zreadtriple.c
    zreadtriple_noheader.c
    zbinary_io.c	
    zreadMM.c
    pzgsequ.c
    pzlaqgs.c
    zldperm_dist.c
    pzlangs.c
    pzutil.c
    pzsymbfact_distdata.c
    zdistribute.c
    pzdistribute.c
    pzgstrf.c
    zstatic_schedule.c
    pzgstrf2.c
    pzgstrs.c
    pzgstrs1.c
    pzgstrs_lsum.c
    pzgstrs_Bglobal.c
    pzgsrfs.c
    pzgsmv.c
    pzgsrfs_ABXglobal.c
    pzgsmv_AXglobal.c
    pzGetDiagU.c
    pzgssvx3d.c   ## 3D code
    znrformat_loc3d.c 
    pzgstrf3d.c
    ztreeFactorization.c 
    ztreeFactorizationGPU.c 
    zscatter3d.c
    zgather.c 
    pz3dcomm.c ztrfAux.c
    zcommunication_aux.c
    ztrfCommWrapper.c
    zsuperlu_blas.c
  )
if (TPL_ENABLE_CUDALIB)
    list(APPEND sources zsuperlu_gpu.cu)
endif()
if (HAVE_COMBBLAS)
    list(APPEND sources z_c2cpp_GetHWPM.cpp zHWPM_CombBLAS.hpp)
endif()
endif() ######### enable compex16

if (TPL_ENABLE_HIPLIB)
  file(GLOB MyFiles *.hip.cpp)
  set_source_files_properties(
    ${MyFiles}
    PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

  set(hipsources
    pdgstrs_lsum_cuda.hip.cpp
    dsuperlu_gpu.hip.cpp
  )
  if(enable_single)
  list(APPEND hipsources ssuperlu_gpu.hip.cpp)
  endif()
  if(enable_complex16)
  list(APPEND hipsources zsuperlu_gpu.hip.cpp)
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

set(superlu_dist_libs ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${BLAS_LIB} ${LAPACK_LIB}
    ${PARMETIS_LIB} ${COMBBLAS_LIB} ${CUDA_LIB} ${HIP_LIB}) 
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
