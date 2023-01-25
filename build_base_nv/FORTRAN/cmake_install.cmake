# Install script for directory: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist_fortran.so.8.1.2"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist_fortran.so.8"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/lib:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libparmetis:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libmetis")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./lib" TYPE SHARED_LIBRARY FILES
    "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN/libsuperlu_dist_fortran.so.8.1.2"
    "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN/libsuperlu_dist_fortran.so.8"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist_fortran.so.8.1.2"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist_fortran.so.8"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/SRC:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libparmetis:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libmetis:"
           NEW_RPATH "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/lib:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libparmetis:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libmetis")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist_fortran.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist_fortran.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist_fortran.so"
         RPATH "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/lib:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libparmetis:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libmetis")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./lib" TYPE SHARED_LIBRARY FILES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN/libsuperlu_dist_fortran.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist_fortran.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist_fortran.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist_fortran.so"
         OLD_RPATH "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/SRC:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libparmetis:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libmetis:"
         NEW_RPATH "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/lib:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libparmetis:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libmetis")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist_fortran.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./lib" TYPE STATIC_LIBRARY FILES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN/libsuperlu_dist_fortran.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN" FILES_MATCHING REGEX "/[^/]*\\.mod$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlu_dist_config.fh")
endif()

