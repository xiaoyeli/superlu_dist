# CMake generated Testfile for 
# Source directory: /gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE
# Build directory: /gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/EXAMPLE
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(pddrive1 "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "4" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/EXAMPLE/pddrive1" "-r" "2" "-c" "2" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/big.rua")
set_tests_properties(pddrive1 PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/CMakeLists.txt;20;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/CMakeLists.txt;44;add_superlu_dist_example;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/CMakeLists.txt;0;")
add_test(pddrive2 "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "4" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/EXAMPLE/pddrive2" "-r" "2" "-c" "2" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/big.rua")
set_tests_properties(pddrive2 PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/CMakeLists.txt;20;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/CMakeLists.txt;49;add_superlu_dist_example;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/CMakeLists.txt;0;")
add_test(pddrive3 "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "4" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/EXAMPLE/pddrive3" "-r" "2" "-c" "2" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/big.rua")
set_tests_properties(pddrive3 PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/CMakeLists.txt;20;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/CMakeLists.txt;54;add_superlu_dist_example;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/CMakeLists.txt;0;")
