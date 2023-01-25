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
CMAKE_BINARY_DIR = /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base

# Include any dependencies generated for this target.
include EXAMPLE/CMakeFiles/psdrive.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include EXAMPLE/CMakeFiles/psdrive.dir/compiler_depend.make

# Include the progress variables for this target.
include EXAMPLE/CMakeFiles/psdrive.dir/progress.make

# Include the compile flags for this target's objects.
include EXAMPLE/CMakeFiles/psdrive.dir/flags.make

EXAMPLE/CMakeFiles/psdrive.dir/psdrive.c.o: EXAMPLE/CMakeFiles/psdrive.dir/flags.make
EXAMPLE/CMakeFiles/psdrive.dir/psdrive.c.o: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/psdrive.c
EXAMPLE/CMakeFiles/psdrive.dir/psdrive.c.o: EXAMPLE/CMakeFiles/psdrive.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object EXAMPLE/CMakeFiles/psdrive.dir/psdrive.c.o"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT EXAMPLE/CMakeFiles/psdrive.dir/psdrive.c.o -MF CMakeFiles/psdrive.dir/psdrive.c.o.d -o CMakeFiles/psdrive.dir/psdrive.c.o -c /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/psdrive.c

EXAMPLE/CMakeFiles/psdrive.dir/psdrive.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/psdrive.dir/psdrive.c.i"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/psdrive.c > CMakeFiles/psdrive.dir/psdrive.c.i

EXAMPLE/CMakeFiles/psdrive.dir/psdrive.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/psdrive.dir/psdrive.c.s"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/psdrive.c -o CMakeFiles/psdrive.dir/psdrive.c.s

EXAMPLE/CMakeFiles/psdrive.dir/screate_matrix.c.o: EXAMPLE/CMakeFiles/psdrive.dir/flags.make
EXAMPLE/CMakeFiles/psdrive.dir/screate_matrix.c.o: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/screate_matrix.c
EXAMPLE/CMakeFiles/psdrive.dir/screate_matrix.c.o: EXAMPLE/CMakeFiles/psdrive.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object EXAMPLE/CMakeFiles/psdrive.dir/screate_matrix.c.o"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT EXAMPLE/CMakeFiles/psdrive.dir/screate_matrix.c.o -MF CMakeFiles/psdrive.dir/screate_matrix.c.o.d -o CMakeFiles/psdrive.dir/screate_matrix.c.o -c /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/screate_matrix.c

EXAMPLE/CMakeFiles/psdrive.dir/screate_matrix.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/psdrive.dir/screate_matrix.c.i"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/screate_matrix.c > CMakeFiles/psdrive.dir/screate_matrix.c.i

EXAMPLE/CMakeFiles/psdrive.dir/screate_matrix.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/psdrive.dir/screate_matrix.c.s"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/screate_matrix.c -o CMakeFiles/psdrive.dir/screate_matrix.c.s

EXAMPLE/CMakeFiles/psdrive.dir/screate_A_x_b.c.o: EXAMPLE/CMakeFiles/psdrive.dir/flags.make
EXAMPLE/CMakeFiles/psdrive.dir/screate_A_x_b.c.o: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/screate_A_x_b.c
EXAMPLE/CMakeFiles/psdrive.dir/screate_A_x_b.c.o: EXAMPLE/CMakeFiles/psdrive.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object EXAMPLE/CMakeFiles/psdrive.dir/screate_A_x_b.c.o"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT EXAMPLE/CMakeFiles/psdrive.dir/screate_A_x_b.c.o -MF CMakeFiles/psdrive.dir/screate_A_x_b.c.o.d -o CMakeFiles/psdrive.dir/screate_A_x_b.c.o -c /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/screate_A_x_b.c

EXAMPLE/CMakeFiles/psdrive.dir/screate_A_x_b.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/psdrive.dir/screate_A_x_b.c.i"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/screate_A_x_b.c > CMakeFiles/psdrive.dir/screate_A_x_b.c.i

EXAMPLE/CMakeFiles/psdrive.dir/screate_A_x_b.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/psdrive.dir/screate_A_x_b.c.s"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base/EXAMPLE && /opt/cray/pe/craype/2.7.19/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE/screate_A_x_b.c -o CMakeFiles/psdrive.dir/screate_A_x_b.c.s

# Object files for target psdrive
psdrive_OBJECTS = \
"CMakeFiles/psdrive.dir/psdrive.c.o" \
"CMakeFiles/psdrive.dir/screate_matrix.c.o" \
"CMakeFiles/psdrive.dir/screate_A_x_b.c.o"

# External object files for target psdrive
psdrive_EXTERNAL_OBJECTS =

EXAMPLE/psdrive: EXAMPLE/CMakeFiles/psdrive.dir/psdrive.c.o
EXAMPLE/psdrive: EXAMPLE/CMakeFiles/psdrive.dir/screate_matrix.c.o
EXAMPLE/psdrive: EXAMPLE/CMakeFiles/psdrive.dir/screate_A_x_b.c.o
EXAMPLE/psdrive: EXAMPLE/CMakeFiles/psdrive.dir/build.make
EXAMPLE/psdrive: SRC/libsuperlu_dist.so.8.1.2
EXAMPLE/psdrive: /opt/cray/pe/libsci/22.11.1.2/nvidia/20.7/x86_64/lib/libsci_nvidia_mp.so
EXAMPLE/psdrive: /global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit//build/Linux-x86_64/libparmetis/libparmetis.so
EXAMPLE/psdrive: /global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit//build/Linux-x86_64/libmetis/libmetis.so
EXAMPLE/psdrive: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/lib64/libcudart.so
EXAMPLE/psdrive: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/profilers/Nsight_Compute/../../math_libs/11.7/lib64/libcusolver.so
EXAMPLE/psdrive: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/profilers/Nsight_Compute/../../math_libs/11.7/lib64/libcublas.so
EXAMPLE/psdrive: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/lib64/libculibos.a
EXAMPLE/psdrive: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/profilers/Nsight_Compute/../../math_libs/11.7/lib64/libcublasLt.so
EXAMPLE/psdrive: /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/profilers/Nsight_Compute/../../math_libs/11.7/lib64/libcusparse.so
EXAMPLE/psdrive: /opt/cray/pe/libsci/22.11.1.2/NVIDIA/20.7/x86_64/lib/libsci_nvidia_mpi_mp.so
EXAMPLE/psdrive: /opt/cray/pe/libsci/22.11.1.2/NVIDIA/20.7/x86_64/lib/libsci_nvidia_mp.so
EXAMPLE/psdrive: EXAMPLE/CMakeFiles/psdrive.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking C executable psdrive"
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base/EXAMPLE && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/psdrive.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
EXAMPLE/CMakeFiles/psdrive.dir/build: EXAMPLE/psdrive
.PHONY : EXAMPLE/CMakeFiles/psdrive.dir/build

EXAMPLE/CMakeFiles/psdrive.dir/clean:
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base/EXAMPLE && $(CMAKE_COMMAND) -P CMakeFiles/psdrive.dir/cmake_clean.cmake
.PHONY : EXAMPLE/CMakeFiles/psdrive.dir/clean

EXAMPLE/CMakeFiles/psdrive.dir/depend:
	cd /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/EXAMPLE /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base/EXAMPLE /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_base/EXAMPLE/CMakeFiles/psdrive.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : EXAMPLE/CMakeFiles/psdrive.dir/depend

