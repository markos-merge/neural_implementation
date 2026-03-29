# CMakeLists.txt
#
# Project: neural_implementation
# C++ Standard: C++20
# Build System: Ninja

cmake_minimum_required(VERSION 3.18)
project(neural_implementation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Release only: strong optimization + SIMD tuned to the *build* machine (Eigen benefits).
# GCC/Clang: -march=native may emit AVX/AVX2; binaries can fail on older CPUs than the builder’s.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	add_compile_options(
		$<$<CONFIG:Release>:-O3>
		$<$<CONFIG:Release>:-march=native>
	)
elseif(MSVC)
	add_compile_options($<$<CONFIG:Release>:/O2>)
endif()

option(NEURAL_ENABLE_CUDA "Enable CUDA language and link CUDA runtime" ON)

set(NEURAL_CUDA_FOUND OFF)
if(NEURAL_ENABLE_CUDA)
	include(CheckLanguage)
	check_language(CUDA)
	if(CMAKE_CUDA_COMPILER)
		enable_language(CUDA)
		set(CMAKE_CUDA_STANDARD 20)
		set(CMAKE_CUDA_STANDARD_REQUIRED ON)
		if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
			# "native" requires CMake >= 3.24; see https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html
			if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
				set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA GPU architectures (semicolon-separated)")
			else()
				set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA GPU architectures (semicolon-separated)")
			endif()
		endif()
		find_package(CUDAToolkit REQUIRED)
		set(NEURAL_CUDA_FOUND ON)
		message(STATUS "CUDA enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
	else()
		message(STATUS "CUDA compiler not found; building without CUDA")
	endif()
endif()

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

# Core library (Tensor is header-only in include/tensors/tensor.hpp)
add_library(neural_impl_lib STATIC)

if(NEURAL_CUDA_FOUND)
	target_sources(neural_impl_lib PRIVATE src/cuda/cuda_runtime.cu src/cuda/cublas_handle.cpp)
	# cublas: SGEMM/DGEMM etc.; cublasLt: mixed precision / FP8 matmul (your custom FP8 path)
	target_link_libraries(neural_impl_lib PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt)
	target_compile_definitions(neural_impl_lib PUBLIC NEURAL_CUDA_ENABLED=1)
else()
	target_sources(neural_impl_lib PRIVATE src/cuda/cuda_runtime_stub.cpp)
	target_compile_definitions(neural_impl_lib PUBLIC NEURAL_CUDA_ENABLED=0)
endif()

# Public headers live under include/; subfolders are added so clients can use short includes
# (e.g. "tensor.hpp", "sequential_nn.hpp", "sgd_optimizer.hpp", "mnist_loader.hpp").
target_include_directories(neural_impl_lib PUBLIC
	include
	include/tensors
	include/layers
	include/losses
	include/cuda
	include/nn
	include/optimizer
	include/app
)
target_link_libraries(neural_impl_lib PUBLIC Eigen3::Eigen)

# Main executable (entry point remains src/main.cpp; demos are under src/app/)
add_executable(neural_implementation src/main.cpp src/app/mnist_demo.cpp src/app/cifar_demo.cpp)
target_link_libraries(neural_implementation PRIVATE neural_impl_lib)

# Tests
add_executable(tests
    tests/test_tensor.cpp
    tests/test_tensor_n.cpp
    tests/test_cuda_tensor.cpp
    tests/test_cuda_nn.cpp
    tests/test_cuda_runtime.cpp
    tests/test_linear_layer.cpp
    tests/test_relu_layer.cpp
    tests/test_sigmoid_layer.cpp
    tests/test_softmax_layer.cpp
    tests/test_loss.cpp
    tests/test_sequential_nn.cpp
    tests/test_sgd_optimizer.cpp
)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain neural_impl_lib)
target_include_directories(tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
include(CTest)
enable_testing()
if(NOT Catch2_FOUND)
	list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
endif()
include(Catch)
catch_discover_tests(tests)
