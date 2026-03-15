# CMakeLists.txt
#
# Project: neural_implementation
# C++ Standard: C++20
# Build System: Ninja

cmake_minimum_required(VERSION 3.14)
project(neural_implementation)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Catch2 for tests (prefer system package to avoid FetchContent rebuilds)
find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
	include(FetchContent)
	set(FETCHCONTENT_UPDATES_DISCONNECTED_Catch2 ON)
	FetchContent_Declare(
	    Catch2
	    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
	    GIT_TAG v3.5.0
	)
	FetchContent_MakeAvailable(Catch2)
endif()

# Core library
add_library(neural_impl_lib
    src/tensor.cpp
)

target_include_directories(neural_impl_lib PUBLIC include)
target_link_libraries(neural_impl_lib PUBLIC Eigen3::Eigen)

# Main executable
add_executable(neural_implementation src/main.cpp)
target_link_libraries(neural_implementation PRIVATE neural_impl_lib)

# Tests
add_executable(tests
    tests/test_tensor.cpp
)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain neural_impl_lib)
include(CTest)
enable_testing()
add_test(NAME tests COMMAND tests)
