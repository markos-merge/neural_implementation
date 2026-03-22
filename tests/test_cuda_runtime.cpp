#include "neural_cuda_runtime.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE( "neural_cuda_device_count is callable", "[cuda][runtime]" )
{
	REQUIRE_NOTHROW( neural::cuda_device_count() );
}

TEST_CASE( "neural_cuda_runtime_ready is callable", "[cuda][runtime]" )
{
	REQUIRE_NOTHROW( neural::cuda_runtime_ready() );
}
