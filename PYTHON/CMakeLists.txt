# include the paths for header files
include_directories(${SuperLU_DIST_SOURCE_DIR}/SRC)
include_directories(${SuperLU_DIST_BINARY_DIR}/PYTHON)

set(sources
    pdbridge.h
    pdbridge.c
    )

# if(enable_complex16)
#   list(APPEND sources pzbridge.h pzbridge.c)
# endif()  

add_library(superlu_dist_python ${sources})
get_target_property(superlu_dist_version superlu_dist VERSION)
get_target_property(superlu_dist_soversion superlu_dist SOVERSION)
set_target_properties(superlu_dist_python PROPERTIES VERSION ${superlu_dist_version})
set_target_properties(superlu_dist_python PROPERTIES SOVERSION ${superlu_dist_soversion})
target_link_libraries(superlu_dist_python superlu_dist)

set(PY_SCRIPT_SOURCE ${CMAKE_SOURCE_DIR}/PYTHON/pddrive.py)
set(PY_SCRIPT_DEST ${CMAKE_BINARY_DIR}/PYTHON/pddrive.py)
# Create a custom command to copy the Python script
add_custom_command(
    OUTPUT ${PY_SCRIPT_DEST}
    COMMAND ${CMAKE_COMMAND} -E copy ${PY_SCRIPT_SOURCE} ${PY_SCRIPT_DEST}
    DEPENDS ${PY_SCRIPT_SOURCE}
    COMMENT "Copying pddrive.py to ${CMAKE_BINARY_DIR}/PYTHON"
)

set(PY_SCRIPT_SOURCE ${CMAKE_SOURCE_DIR}/PYTHON/pdbridge.py)
set(PY_SCRIPT_DEST ${CMAKE_BINARY_DIR}/PYTHON/pdbridge.py)
# Create a custom command to copy the Python script
add_custom_command(
    OUTPUT ${PY_SCRIPT_DEST}
    COMMAND ${CMAKE_COMMAND} -E copy ${PY_SCRIPT_SOURCE} ${PY_SCRIPT_DEST}
    DEPENDS ${PY_SCRIPT_SOURCE}
    COMMENT "Copying pdbridge.py to ${CMAKE_BINARY_DIR}/PYTHON"
)

# Create a custom target named 'python' that depends on the output file
add_custom_target(python
    DEPENDS ${PY_SCRIPT_DEST}
)

add_dependencies(python superlu_dist_python)

install(TARGETS superlu_dist_python
# DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
)
install(FILES pddrive.py DESTINATION "${INSTALL_LIB_DIR}/PYTHON")
install(FILES pdbridge.py DESTINATION "${INSTALL_LIB_DIR}/PYTHON")







