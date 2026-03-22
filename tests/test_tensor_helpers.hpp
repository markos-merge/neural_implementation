#ifndef NEURAL_IMPL_TEST_TENSOR_HELPERS_HPP
#define NEURAL_IMPL_TEST_TENSOR_HELPERS_HPP

#include "tensor.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace neural::test {

/// Default tolerances for float32 reference vs. GPU (adjust for FP16 / TF32 later).
inline constexpr float k_fp32_abs_tol = 1e-5f;
inline constexpr float k_fp32_rel_tol = 1e-4f;

/// Maximum absolute difference over all elements (useful for diagnostics).
template <typename T>
T max_abs_diff( Tensor<T> const &a, Tensor<T> const &b )
{
	REQUIRE( a.rows() == b.rows() );
	REQUIRE( a.cols() == b.cols() );
	T m = 0;
	for ( std::size_t i = 0; i < a.rows(); ++i )
		for ( std::size_t j = 0; j < a.cols(); ++j )
			m = std::max( m, std::abs( a( i, j ) - b( i, j ) ) );
	return m;
}

/// Element-wise check: |a-b| <= abs_tol + rel_tol * max(1, |a|, |b|).
template <typename T>
void require_tensor_close( Tensor<T> const &a, Tensor<T> const &b,
                           T abs_tol = k_fp32_abs_tol,
                           T rel_tol = k_fp32_rel_tol )
{
	REQUIRE( a.rows() == b.rows() );
	REQUIRE( a.cols() == b.cols() );
	using Catch::Matchers::WithinAbs;
	for ( std::size_t i = 0; i < a.rows(); ++i ) {
		for ( std::size_t j = 0; j < a.cols(); ++j ) {
			T const scale =
			    std::max<T>( { static_cast<T>( 1 ), std::abs( a( i, j ) ),
			                   std::abs( b( i, j ) ) } );
			T const tol = abs_tol + rel_tol * scale;
			REQUIRE_THAT( a( i, j ), WithinAbs( b( i, j ), tol ) );
		}
	}
}

} // namespace neural::test

#endif
