# include the paths for header files
include_directories(${SuperLU_DIST_SOURCE_DIR}/SRC)
include_directories(${SuperLU_DIST_BINARY_DIR}/FORTRAN)

set(mod_sources superlu_mod.f90 superlupara.f90)
set(sources
    superlu_c2f_wrap.c  # initialize precision-independent file
    superlupara.f90
    superlu_mod.f90
    )

if(enable_double)
  list(APPEND sources c2f_dcreate_matrix_x_b.c superlu_c2f_dwrap.c)
endif()
if(enable_complex16)
  list(APPEND sources c2f_zcreate_matrix_x_b.c superlu_c2f_zwrap.c)
endif()  

# see https://gitlab.kitware.com/cmake/cmake/-/issues/17525, why we need to separate MOD files from other source files
add_library(superlu_dist_fortran_mod OBJECT ${mod_sources})
add_library(superlu_dist_fortran $<TARGET_OBJECTS:superlu_dist_fortran_mod> ${sources})
add_library(superlu_dist_fortran-static STATIC $<TARGET_OBJECTS:superlu_dist_fortran_mod> ${sources})
# set(targets superlu_dist_fortran)
get_target_property(superlu_dist_version superlu_dist VERSION)
get_target_property(superlu_dist_soversion superlu_dist SOVERSION)
set_target_properties(superlu_dist_fortran PROPERTIES VERSION ${superlu_dist_version})
set_target_properties(superlu_dist_fortran PROPERTIES SOVERSION ${superlu_dist_soversion})
target_link_libraries(superlu_dist_fortran superlu_dist)

set_target_properties(superlu_dist_fortran-static PROPERTIES OUTPUT_NAME superlu_dist_fortran)
set_target_properties(superlu_dist_fortran-static PROPERTIES VERSION ${superlu_dist_version})
set_target_properties(superlu_dist_fortran-static PROPERTIES SOVERSION ${superlu_dist_soversion})
target_link_libraries(superlu_dist_fortran-static superlu_dist)

# depends on FPP defs
add_dependencies(superlu_dist_fortran config_f)
add_dependencies(superlu_dist_fortran-static config_f)

install(TARGETS superlu_dist_fortran
# DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
)
install(TARGETS superlu_dist_fortran-static
# DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN *.mod
  )
install(FILES superlu_dist_config.fh
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

# Fortran MPI stuff
add_definitions(${MPI_Fortran_COMPILE_FLAGS})
include_directories(${MPI_Fortran_INCLUDE_PATH})
link_directories(${MPI_Fortran_LIBRARIES})

# Libs to be linked with the Fortran codes
set(fortran_link_libs superlu_dist_fortran ${MPI_Fortran_LIBRARIES} ${BLAS_LIB} ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
set(fortran_link_libs superlu_dist_fortran-static ${MPI_Fortran_LIBRARIES} ${BLAS_LIB} ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
set(all_link_libs ${fortran_link_libs} superlu_dist)

#message("!!! in Fortran: MPI_Fortran_LINK_FLAGS='${MPI_Fortran_LINK_FLAGS}'")
#message("!!! in Fortran: all_link_libs='${all_link_libs}'")
#message("!!! in Fortran: cxx_implicit='${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES}'")
if (NOT MSVC)
  list(APPEND all_link_libs m)
endif ()

if(enable_double)
## if(FALSE)
  set(F_DEXM ${F_MOD} f_pddrive.F90)
  add_executable(f_pddrive ${F_DEXM})
  target_link_libraries(f_pddrive ${all_link_libs})
  # set_target_properties(f_pddrive PROPERTIES LINKER_LANGUAGE Fortran CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_target_properties(f_pddrive PROPERTIES LINKER_LANGUAGE CXX LINK_FLAGS "${MPI_Fortran_LINK_FLAGS}")
  
  set(F_DEXM3D f_pddrive3d.F90)
  add_executable(f_pddrive3d ${F_DEXM3D})
  target_link_libraries(f_pddrive3d ${all_link_libs})
  set_target_properties(f_pddrive3d PROPERTIES LINKER_LANGUAGE CXX LINK_FLAGS "${MPI_Fortran_LINK_FLAGS}")
  
  set(F_5x5 f_5x5.F90 sp_ienv.c)
  add_executable(f_5x5 ${F_5x5})
  target_link_libraries(f_5x5 ${all_link_libs})
  set_target_properties(f_5x5 PROPERTIES LINKER_LANGUAGE CXX LINK_FLAGS "${MPI_Fortran_LINK_FLAGS}")
  
endif()

##if(FALSE)
if(enable_complex16)
  set(F_ZEXM f_pzdrive.F90)
  add_executable(f_pzdrive ${F_ZEXM})
  target_link_libraries(f_pzdrive ${all_link_libs})
#  set_target_properties(f_pzdrive PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(f_pzdrive PROPERTIES LINKER_LANGUAGE CXX LINK_FLAGS "${MPI_Fortran_LINK_FLAGS}")

  set(F_ZEXM3D f_pzdrive3d.F90)
  add_executable(f_pzdrive3d ${F_ZEXM3D})
  target_link_libraries(f_pzdrive3d ${all_link_libs})
  set_target_properties(f_pzdrive3d PROPERTIES LINKER_LANGUAGE CXX LINK_FLAGS "${MPI_Fortran_LINK_FLAGS}")
  
endif()


# Format superlu_dist_config.fh from superlu_dist_config.h in C
add_custom_command(
  OUTPUT superlu_dist_config.fh
  COMMAND sed;'/^\\//;d';<;superlu_dist_config.h;>;superlu_dist_config.fh
  COMMAND cp;superlu_dist_config.fh;${SuperLU_DIST_SOURCE_DIR}/FORTRAN/.
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/FORTRAN
)
add_custom_target(config_f DEPENDS superlu_dist_config.fh)
