# CMake generated Testfile for 
# Source directory: /gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST
# Build directory: /gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(pdtest_1x1_1_2_8_20_SP "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "1" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST/pdtest" "-r" "1" "-c" "1" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_1x1_1_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;47;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_1x1_3_2_8_20_SP "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "1" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST/pdtest" "-r" "1" "-c" "1" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_1x1_3_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;47;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_1x2_1_2_8_20_SP "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "2" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST/pdtest" "-r" "1" "-c" "2" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_1x2_1_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;47;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_1x2_3_2_8_20_SP "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "2" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST/pdtest" "-r" "1" "-c" "2" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_1x2_3_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;47;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_1x3_1_2_8_20_SP "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "3" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST/pdtest" "-r" "1" "-c" "3" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_1x3_1_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;47;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_1x3_3_2_8_20_SP "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "3" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST/pdtest" "-r" "1" "-c" "3" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_1x3_3_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;47;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_2x1_1_2_8_20_SP "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "2" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST/pdtest" "-r" "2" "-c" "1" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_2x1_1_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;47;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_2x1_3_2_8_20_SP "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "2" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST/pdtest" "-r" "2" "-c" "1" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_2x1_3_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;47;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_2x2_1_2_8_20_SP "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "4" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST/pdtest" "-r" "2" "-c" "2" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_2x2_1_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;47;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_2x2_3_2_8_20_SP "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "4" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST/pdtest" "-r" "2" "-c" "2" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_2x2_3_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;47;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_2x3_1_2_8_20_SP "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "6" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST/pdtest" "-r" "2" "-c" "3" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_2x3_1_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;47;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_2x3_3_2_8_20_SP "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "6" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST/pdtest" "-r" "2" "-c" "3" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_2x3_3_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;47;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_5x1_1_2_8_20_SP "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "5" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST/pdtest" "-r" "5" "-c" "1" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_5x1_1_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;47;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_5x1_3_2_8_20_SP "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "5" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST/pdtest" "-r" "5" "-c" "1" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_5x1_3_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;47;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_5x2_1_2_8_20_SP "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "10" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST/pdtest" "-r" "5" "-c" "2" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_5x2_1_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;47;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_5x2_3_2_8_20_SP "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "10" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST/pdtest" "-r" "5" "-c" "2" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_5x2_3_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;47;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_5x3_1_2_8_20_SP "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "15" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST/pdtest" "-r" "5" "-c" "3" "-s" "1" "-b" "2" "-x" "8" "-m" "20" "-f" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_5x3_1_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;47;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;0;")
add_test(pdtest_5x3_3_2_8_20_SP "/soft/restricted/CNDA/mpich/drop39.2/mpich-ofi-sockets-icc-default-gen9-drop39/bin/mpiexec" "-n" "15" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/TEST/pdtest" "-r" "5" "-c" "3" "-s" "3" "-b" "2" "-x" "8" "-m" "20" "-f" "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/EXAMPLE/g20.rua")
set_tests_properties(pdtest_5x3_3_2_8_20_SP PROPERTIES  _BACKTRACE_TRIPLES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;47;add_test;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;77;add_superlu_dist_tests;/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/TEST/CMakeLists.txt;0;")
