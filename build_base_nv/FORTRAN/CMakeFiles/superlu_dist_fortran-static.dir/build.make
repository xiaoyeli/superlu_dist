# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /global/common/software/nersc/pm-2022q4/spack/linux-sles15-zen/cmake-3.24.3-k5msymx/bin/cmake

# The command to remove a file.
RM = /global/common/software/nersc/pm-2022q4/spack/linux-sles15-zen/cmake-3.24.3-k5msymx/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv

# Include any dependencies generated for this target.
include FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/compiler_depend.make

# Include the progress variables for this target.
include FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/progress.make

# Include the compile flags for this target's objects.
include FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/flags.make

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_wrap.c.o: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/flags.make
FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_wrap.c.o: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlu_c2f_wrap.c
FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_wrap.c.o: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_wrap.c.o"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_wrap.c.o -MF CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_wrap.c.o.d -o CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_wrap.c.o -c /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlu_c2f_wrap.c

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_wrap.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_wrap.c.i"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlu_c2f_wrap.c > CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_wrap.c.i

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_wrap.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_wrap.c.s"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlu_c2f_wrap.c -o CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_wrap.c.s

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlupara.f90.o: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/flags.make
FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlupara.f90.o: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlupara.f90
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building Fortran object FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlupara.f90.o"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/ftn $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -Mfreeform -Mpreprocess -c /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlupara.f90 -o CMakeFiles/superlu_dist_fortran-static.dir/superlupara.f90.o

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlupara.f90.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/superlu_dist_fortran-static.dir/superlupara.f90.i"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/ftn $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -Mfreeform -Mpreprocess -E /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlupara.f90 > CMakeFiles/superlu_dist_fortran-static.dir/superlupara.f90.i

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlupara.f90.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/superlu_dist_fortran-static.dir/superlupara.f90.s"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/ftn $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -Mfreeform -Mpreprocess -S /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlupara.f90 -o CMakeFiles/superlu_dist_fortran-static.dir/superlupara.f90.s

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_mod.f90.o: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/flags.make
FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_mod.f90.o: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlu_mod.f90
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building Fortran object FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_mod.f90.o"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/ftn $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -Mfreeform -Mpreprocess -c /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlu_mod.f90 -o CMakeFiles/superlu_dist_fortran-static.dir/superlu_mod.f90.o

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_mod.f90.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/superlu_dist_fortran-static.dir/superlu_mod.f90.i"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/ftn $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -Mfreeform -Mpreprocess -E /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlu_mod.f90 > CMakeFiles/superlu_dist_fortran-static.dir/superlu_mod.f90.i

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_mod.f90.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/superlu_dist_fortran-static.dir/superlu_mod.f90.s"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/ftn $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -Mfreeform -Mpreprocess -S /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlu_mod.f90 -o CMakeFiles/superlu_dist_fortran-static.dir/superlu_mod.f90.s

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/c2f_dcreate_matrix_x_b.c.o: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/flags.make
FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/c2f_dcreate_matrix_x_b.c.o: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/c2f_dcreate_matrix_x_b.c
FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/c2f_dcreate_matrix_x_b.c.o: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/c2f_dcreate_matrix_x_b.c.o"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/c2f_dcreate_matrix_x_b.c.o -MF CMakeFiles/superlu_dist_fortran-static.dir/c2f_dcreate_matrix_x_b.c.o.d -o CMakeFiles/superlu_dist_fortran-static.dir/c2f_dcreate_matrix_x_b.c.o -c /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/c2f_dcreate_matrix_x_b.c

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/c2f_dcreate_matrix_x_b.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/superlu_dist_fortran-static.dir/c2f_dcreate_matrix_x_b.c.i"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/c2f_dcreate_matrix_x_b.c > CMakeFiles/superlu_dist_fortran-static.dir/c2f_dcreate_matrix_x_b.c.i

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/c2f_dcreate_matrix_x_b.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/superlu_dist_fortran-static.dir/c2f_dcreate_matrix_x_b.c.s"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/c2f_dcreate_matrix_x_b.c -o CMakeFiles/superlu_dist_fortran-static.dir/c2f_dcreate_matrix_x_b.c.s

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_dwrap.c.o: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/flags.make
FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_dwrap.c.o: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlu_c2f_dwrap.c
FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_dwrap.c.o: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_dwrap.c.o"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_dwrap.c.o -MF CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_dwrap.c.o.d -o CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_dwrap.c.o -c /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlu_c2f_dwrap.c

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_dwrap.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_dwrap.c.i"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlu_c2f_dwrap.c > CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_dwrap.c.i

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_dwrap.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_dwrap.c.s"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlu_c2f_dwrap.c -o CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_dwrap.c.s

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/c2f_zcreate_matrix_x_b.c.o: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/flags.make
FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/c2f_zcreate_matrix_x_b.c.o: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/c2f_zcreate_matrix_x_b.c
FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/c2f_zcreate_matrix_x_b.c.o: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/c2f_zcreate_matrix_x_b.c.o"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/c2f_zcreate_matrix_x_b.c.o -MF CMakeFiles/superlu_dist_fortran-static.dir/c2f_zcreate_matrix_x_b.c.o.d -o CMakeFiles/superlu_dist_fortran-static.dir/c2f_zcreate_matrix_x_b.c.o -c /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/c2f_zcreate_matrix_x_b.c

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/c2f_zcreate_matrix_x_b.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/superlu_dist_fortran-static.dir/c2f_zcreate_matrix_x_b.c.i"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/c2f_zcreate_matrix_x_b.c > CMakeFiles/superlu_dist_fortran-static.dir/c2f_zcreate_matrix_x_b.c.i

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/c2f_zcreate_matrix_x_b.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/superlu_dist_fortran-static.dir/c2f_zcreate_matrix_x_b.c.s"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/c2f_zcreate_matrix_x_b.c -o CMakeFiles/superlu_dist_fortran-static.dir/c2f_zcreate_matrix_x_b.c.s

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_zwrap.c.o: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/flags.make
FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_zwrap.c.o: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlu_c2f_zwrap.c
FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_zwrap.c.o: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_zwrap.c.o"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_zwrap.c.o -MF CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_zwrap.c.o.d -o CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_zwrap.c.o -c /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlu_c2f_zwrap.c

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_zwrap.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_zwrap.c.i"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlu_c2f_zwrap.c > CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_zwrap.c.i

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_zwrap.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_zwrap.c.s"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN/superlu_c2f_zwrap.c -o CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_zwrap.c.s

# Object files for target superlu_dist_fortran-static
superlu_dist_fortran__static_OBJECTS = \
"CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_wrap.c.o" \
"CMakeFiles/superlu_dist_fortran-static.dir/superlupara.f90.o" \
"CMakeFiles/superlu_dist_fortran-static.dir/superlu_mod.f90.o" \
"CMakeFiles/superlu_dist_fortran-static.dir/c2f_dcreate_matrix_x_b.c.o" \
"CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_dwrap.c.o" \
"CMakeFiles/superlu_dist_fortran-static.dir/c2f_zcreate_matrix_x_b.c.o" \
"CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_zwrap.c.o"

# External object files for target superlu_dist_fortran-static
superlu_dist_fortran__static_EXTERNAL_OBJECTS =

FORTRAN/libsuperlu_dist_fortran.a: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_wrap.c.o
FORTRAN/libsuperlu_dist_fortran.a: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlupara.f90.o
FORTRAN/libsuperlu_dist_fortran.a: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_mod.f90.o
FORTRAN/libsuperlu_dist_fortran.a: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/c2f_dcreate_matrix_x_b.c.o
FORTRAN/libsuperlu_dist_fortran.a: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_dwrap.c.o
FORTRAN/libsuperlu_dist_fortran.a: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/c2f_zcreate_matrix_x_b.c.o
FORTRAN/libsuperlu_dist_fortran.a: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/superlu_c2f_zwrap.c.o
FORTRAN/libsuperlu_dist_fortran.a: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/build.make
FORTRAN/libsuperlu_dist_fortran.a: FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking Fortran static library libsuperlu_dist_fortran.a"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && $(CMAKE_COMMAND) -P CMakeFiles/superlu_dist_fortran-static.dir/cmake_clean_target.cmake
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/superlu_dist_fortran-static.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/build: FORTRAN/libsuperlu_dist_fortran.a
.PHONY : FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/build

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/clean:
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN && $(CMAKE_COMMAND) -P CMakeFiles/superlu_dist_fortran-static.dir/cmake_clean.cmake
.PHONY : FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/clean

FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/depend:
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/FORTRAN /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base_nv/FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : FORTRAN/CMakeFiles/superlu_dist_fortran-static.dir/depend

