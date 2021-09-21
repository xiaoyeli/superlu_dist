set(headers
    superlu_FCnames.h
    gpublas_utils.h
    dcomplex.h
    machines.h
    psymbfact.h
    superlu_defs.h
    superlu_enum_consts.h
    supermatrix.h
    util_dist.h
    colamd.h
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
  gpublas_utils.c
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
)
if (MSVC)
  list(APPEND sources wingetopt.c)
endif ()


set_source_files_properties(superlu_timer.c PROPERTIES COMPILE_FLAGS -O0)

if(enable_double)
  list(APPEND headers superlu_ddefs.h)

if(TPL_ENABLE_CUDALIB)
  list(APPEND sources pdgstrs_lsum_cuda.cu)
endif()
# if(TPL_ENABLE_HIPLIB)
#   list(APPEND sources pdgstrs_lsum_cuda.hip.cpp)
# endif()    	

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
  )
if (HAVE_COMBBLAS)
  list(APPEND sources d_c2cpp_GetHWPM.cpp dHWPM_CombBLAS.hpp)
endif()

endif() ## enable double


if(enable_complex16)
  list(APPEND headers superlu_zdefs.h)

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
  )
if (HAVE_COMBBLAS)
  list(APPEND sources z_c2cpp_GetHWPM.cpp zHWPM_CombBLAS.hpp)
endif()
endif()

if (TPL_ENABLE_HIPLIB)
  set_source_files_properties(
    pdgstrs_lsum_cuda.hip.cpp
    PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  hip_add_library(superlu_dist "pdgstrs_lsum_cuda.hip.cpp")
else()
  add_library(superlu_dist "")
endif()
# if (BUILD_SHARED_LIBS)
#   set_property(TARGET superlu_dist PROPERTY POSITION_INDEPENDENT_CODE ON)
# endif()
    
if (BUILD_SHARED_LIBS AND BUILD_STATIC_LIBS)
if (TPL_ENABLE_HIPLIB)
  hip_add_library(superlu_dist-static STATIC "pdgstrs_lsum_cuda.hip.cpp")
else()
  add_library(superlu_dist-static STATIC "")
endif()
endif()
# if (BUILD_SHARED_LIBS)
#   set_property(TARGET superlu_dist-static PROPERTY POSITION_INDEPENDENT_CODE ON)
# endif()

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
