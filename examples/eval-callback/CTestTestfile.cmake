# CMake generated Testfile for 
# Source directory: /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github/examples/eval-callback
# Build directory: /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github/examples/eval-callback
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[test-eval-callback-download-model]=] "/usr/bin/cmake" "-DDEST=/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github/tinyllamas/stories15M-q4_0.gguf" "-DNAME=tinyllamas/stories15M-q4_0.gguf" "-DHASH=SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739" "-P" "/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github/cmake/download-models.cmake")
set_tests_properties([=[test-eval-callback-download-model]=] PROPERTIES  FIXTURES_SETUP "test-eval-callback-download-model" _BACKTRACE_TRIPLES "/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github/examples/eval-callback/CMakeLists.txt;17;add_test;/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github/examples/eval-callback/CMakeLists.txt;0;")
add_test([=[test-eval-callback]=] "/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github/bin/llama-eval-callback" "-m" "/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github/tinyllamas/stories15M-q4_0.gguf" "--prompt" "hello" "--seed" "42" "-ngl" "0")
set_tests_properties([=[test-eval-callback]=] PROPERTIES  FIXTURES_REQUIRED "test-eval-callback-download-model" _BACKTRACE_TRIPLES "/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github/examples/eval-callback/CMakeLists.txt;24;add_test;/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github/examples/eval-callback/CMakeLists.txt;0;")
