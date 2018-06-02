set(headers
    Cnames.h
    cublas_utils.h
    dcomplex.h
    machines.h
    psymbfact.h
    superlu_defs.h
    superlu_enum_consts.h
    supermatrix.h
    util_dist.h
    colamd.h
    ${CMAKE_CURRENT_BINARY_DIR}/superlu_dist_config.h
)
if (MSVC)
  list(APPEND headers wingetopt.h)
endif ()

# first: precision-independent files
set(sources
  sp_ienv.c
  etree.c 
  sp_colorder.c
  get_perm_c.c
  mmd.c
  comm.c
  memory.c
  util.c
  superlu_grid.c
  pxerr_dist.c
  superlu_timer.c
  symbfact.c
  psymbfact.c
  psymbfact_util.c
  get_perm_c_parmetis.c
  mc64ad_dist.c
  static_schedule.c
  xerr_dist.c
  smach_dist.c
  dmach_dist.c
  colamd.c
  superlu_dist_version.c
)
if (MSVC)
  list(APPEND sources wingetopt.c)
endif ()

if (HAVE_COMBBLAS)
  list(APPEND sources c2cpp_GetAWPM.cpp AWPM_CombBLAS.hpp)
endif()

set_source_files_properties(superlu_timer.c PROPERTIES COMPILE_FLAGS -O0)

if(enable_double)
  list(APPEND headers superlu_ddefs.h)

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
endif()

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
endif()

add_library(superlu_dist ${sources} ${HEADERS})
set(targets superlu_dist)
if (BUILD_SHARED_LIBS AND BUILD_STATIC_LIBS)
  # build both shared and static libs
  add_library(superlu_dist-static STATIC ${sources} ${HEADERS})
  list(APPEND targets superlu_dist-static)
endif()

set(superlu_dist_libs ${MPI_C_LIBRARIES} ${BLAS_LIB} 
    ${PARMETIS_LIB} ${COMBBLAS_LIB})
if (NOT MSVC)
  list(APPEND superlu_dist_libs m)
endif ()

foreach(target ${targets})
    target_link_libraries(${target} ${superlu_dist_libs})
    if (HAVE_COMBBLAS)
        set_target_properties(${target} PROPERTIES
                              OUTPUT_NAME superlu_dist
                              VERSION ${PROJECT_VERSION}
                              SOVERSION ${VERSION_MAJOR}
			      LINKER_LANGUAGE CXX
			      )
    else()
        set_target_properties(${target} PROPERTIES
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
