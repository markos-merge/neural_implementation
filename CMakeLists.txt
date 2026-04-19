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

# Release: -O3 + native tuning. RelWithDebInfo: CMake supplies -O2 -g -DNDEBUG; we add the same
# -march=native as Release and -fno-omit-frame-pointer so profilers (e.g. perf) get usable stacks.
# GCC/Clang: -march=native may emit AVX/AVX2; binaries can fail on older CPUs than the builder’s.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	add_compile_options(
		$<$<CONFIG:Release>:-O3>
		$<$<CONFIG:Release>:-march=native>
		$<$<CONFIG:RelWithDebInfo>:-march=native>
		$<$<CONFIG:RelWithDebInfo>:-fno-omit-frame-pointer>
	)
elseif(MSVC)
	add_compile_options(
		$<$<CONFIG:Release>:/O2>
		$<$<CONFIG:RelWithDebInfo>:/O2>
	)
endif()

# Use mold linker for faster link times (GCC/Clang only).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	add_link_options(-fuse-ld=mold)
endif()

option(NEURAL_ENABLE_CUDA "Enable CUDA language and link CUDA runtime" OFF)

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

# Eigen (system-installed 5.x under /usr/share/eigen3/cmake)
find_package(Eigen3 REQUIRED NO_MODULE
	HINTS /usr/share/eigen3/cmake)

find_package( OpenMP )

find_package(OpenCV REQUIRED)

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
target_sources(neural_impl_lib PRIVATE src/image_processing/image_augmentation.cpp)

if(NEURAL_CUDA_FOUND)
	target_sources(neural_impl_lib PRIVATE src/cuda/cuda_runtime.cu src/cuda/cublas_handle.cpp)
	# cublas: SGEMM/DGEMM etc.; cublasLt: mixed precision / FP8 matmul (your custom FP8 path)
	target_link_libraries(neural_impl_lib PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt)
	target_compile_definitions(neural_impl_lib PUBLIC NEURAL_CUDA_ENABLED=1)

	# Collect extra hints from pip-installed nvidia/cudnn packages (VIRTUAL_ENV or ~/.local).
	set(_cudnn_extra_include_hints "")
	set(_cudnn_extra_lib_hints "")
	foreach(_base "$ENV{VIRTUAL_ENV}" "$ENV{HOME}/.local")
		if(NOT _base STREQUAL "")
			file(GLOB _pip_dirs "${_base}/lib/python*/site-packages/nvidia/cudnn")
			foreach(_d IN LISTS _pip_dirs)
				list(APPEND _cudnn_extra_include_hints "${_d}/include")
				list(APPEND _cudnn_extra_lib_hints     "${_d}/lib")
			endforeach()
		endif()
	endforeach()

	find_path(CUDNN_INCLUDE_DIR cudnn.h
		HINTS
			${CUDNN_ROOT}/include
			$ENV{CUDNN_ROOT}/include
			${_cudnn_extra_include_hints}
			/usr/include
			/usr/local/cuda/include
			/opt/cuda/include)
	find_library(CUDNN_LIBRARY
		NAMES cudnn cudnn.so.9 cudnn.so.8
		HINTS
			${CUDNN_ROOT}/lib
			$ENV{CUDNN_ROOT}/lib
			${_cudnn_extra_lib_hints}
			/usr/lib
			/usr/lib64
			/usr/local/cuda/lib64
			/opt/cuda/lib64)

	if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
		target_include_directories(neural_impl_lib PUBLIC ${CUDNN_INCLUDE_DIR})
		target_link_libraries(neural_impl_lib PUBLIC ${CUDNN_LIBRARY})
		target_sources(neural_impl_lib PRIVATE src/cuda/cudnn_handle.cpp)
		message(STATUS "cuDNN found: ${CUDNN_LIBRARY} (headers: ${CUDNN_INCLUDE_DIR})")
	else()
		message(FATAL_ERROR "CUDA is enabled but cuDNN was not found. Set CUDNN_ROOT to the cuDNN installation prefix.")
	endif()
	target_compile_definitions(neural_impl_lib PUBLIC NEURAL_CUDNN_ENABLED=1)
else()
	target_sources(neural_impl_lib PRIVATE src/cuda/cuda_runtime_stub.cpp)
	target_compile_definitions(neural_impl_lib PUBLIC NEURAL_CUDA_ENABLED=0)
	target_compile_definitions(neural_impl_lib PUBLIC NEURAL_CUDNN_ENABLED=0)
endif()

# Public headers live under include/ and image_processing/; subfolders are added so clients
# can use short includes (e.g. "tensor.hpp", "image_augmentation.hpp", "mnist_loader.hpp").
target_include_directories(neural_impl_lib PUBLIC
	include
	include/tensors
	include/layers
	include/losses
	include/cuda
	include/nn
	include/optimizer
	include/app
	image_processing
)
target_link_libraries(neural_impl_lib PUBLIC Eigen3::Eigen ${OpenCV_LIBS})
target_include_directories(neural_impl_lib PUBLIC ${OpenCV_INCLUDE_DIRS})

option(ENABLE_OPENMP "Enable OpenMP" ON)
if( ENABLE_OPENMP AND OpenMP_CXX_FOUND )
	message( STATUS "OPENMP enabled" )
	target_link_libraries( neural_impl_lib PUBLIC OpenMP::OpenMP_CXX )
else()
	message( STATUS "OPENMP DISABLED" )
endif()

# Main executable (entry point remains src/main.cpp; demos are under src/app/)
add_executable(neural_implementation src/main.cpp src/app/mnist_demo.cpp src/app/cifar_demo.cpp src/app/conv_demo.cpp src/app/conv_demo_momentum.cpp src/app/conv_demo_cuda.cpp)
# target_include_directories(neural_implementation PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(neural_implementation PRIVATE neural_impl_lib)
# target_link_libraries(neural_implementation PRIVATE neural_impl_lib)

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
    tests/test_conv_backward.cpp
    tests/test_max_pool_layer.cpp
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
