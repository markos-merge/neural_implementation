#include "tensor_n.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <array>
#include <numeric>
#include <vector>

using neural::TensorN;
using Catch::Matchers::WithinAbs;

template <std::size_t N>
static std::array<std::size_t, N> ix( std::initializer_list<std::size_t> il )
{
	std::array<std::size_t, N> a{};
	std::size_t i = 0;
	for ( auto v : il ) a[i++] = v;
	return a;
}

namespace {

constexpr float eps = 1e-5f;

struct BackwardResult {
	TensorN<4> grad_weights;
	TensorN<1> grad_bias;
	TensorN<4> grad_output;
};

BackwardResult run_backward( TensorN<4> const &kernel,
                             TensorN<2> const &im2col,
                             std::array<std::size_t, 4> const &in_shape,
                             std::array<std::size_t, 4> const &out_shape )
{
	TensorN<4> grad_in( out_shape, 1.f );

	TensorN<4> grad_transposed;
	grad_in.swapAxes( ix<4>({ 1, 0, 2, 3 }), grad_transposed );

	auto const ws = kernel.shape();

	TensorN<4> grad_weights( ws );
	grad_transposed.multiply( im2col, true, grad_weights );

	TensorN<1> grad_bias( { ws[0] } );
	grad_transposed.reduceSumToDim( 0, grad_bias );

	auto const gs = grad_transposed.shape();
	TensorN<2> grad_2d(
	    { gs[0], gs[1] * gs[2] * gs[3] },
	    grad_transposed.data(),
	    grad_transposed.data() + grad_transposed.size() );

	TensorN<4> wt;
	TensorN<4> kernel_copy = kernel;
	kernel_copy.swapAxes( ix<4>({ 1, 2, 3, 0 }), wt );
	wt.reshape( ix<4>({ ws[1] * ws[2] * ws[3], ws[0], 1, 1 }) );

	TensorN<4> grad_col( ix<4>({ ws[1] * ws[2] * ws[3], grad_2d.shape()[1], 1, 1 }) );
	wt.multiply( grad_2d, false, grad_col );

	TensorN<4> grad_output( in_shape, 0.f );
	grad_col.col2Im( ws, in_shape, grad_output );

	return { std::move( grad_weights ), std::move( grad_bias ), std::move( grad_output ) };
}

} // namespace

// --------------------------------------------------------------------------
// swapAxes
// --------------------------------------------------------------------------

TEST_CASE( "swapAxes transposes first two axes of a 4D tensor", "[tensor_n][swapAxes]" )
{
	std::vector<float> data{ 1, 2, 3, 4, 5, 6 };
	TensorN<4> t( data, ix<4>({ 2, 3, 1, 1 }) );

	TensorN<4> out;
	t.swapAxes( ix<4>({ 1, 0, 2, 3 }), out );

	REQUIRE( out.shape() == ix<4>({ 3, 2, 1, 1 }) );
	CHECK_THAT( out( ix<4>({ 0, 0, 0, 0 }) ), WithinAbs( 1.f, eps ) );
	CHECK_THAT( out( ix<4>({ 0, 1, 0, 0 }) ), WithinAbs( 4.f, eps ) );
	CHECK_THAT( out( ix<4>({ 1, 0, 0, 0 }) ), WithinAbs( 2.f, eps ) );
	CHECK_THAT( out( ix<4>({ 1, 1, 0, 0 }) ), WithinAbs( 5.f, eps ) );
	CHECK_THAT( out( ix<4>({ 2, 0, 0, 0 }) ), WithinAbs( 3.f, eps ) );
	CHECK_THAT( out( ix<4>({ 2, 1, 0, 0 }) ), WithinAbs( 6.f, eps ) );
}

TEST_CASE( "swapAxes with identity permutation is a copy", "[tensor_n][swapAxes]" )
{
	std::vector<float> data{ 1, 2, 3, 4 };
	TensorN<4> t( data, ix<4>({ 1, 1, 2, 2 }) );

	TensorN<4> out;
	t.swapAxes( ix<4>({ 0, 1, 2, 3 }), out );

	REQUIRE( out.shape() == t.shape() );
	for ( std::size_t i = 0; i < 4; ++i )
		CHECK_THAT( out.data()[i], WithinAbs( data[i], eps ) );
}

// --------------------------------------------------------------------------
// reduceSumToDim
// --------------------------------------------------------------------------

TEST_CASE( "reduceSumToDim: 4D tensor along axis 0", "[tensor_n][reduce_sum]" )
{
	std::vector<float> const data{ 1, 2, 3, 10, 20, 30 };
	TensorN<4> t( data, ix<4>({ 2, 3, 1, 1 }) );

	TensorN<1> out;
	t.reduceSumToDim( 0, out );

	REQUIRE( out.shape()[0] == 2 );
	CHECK_THAT( out( ix<1>({ 0 }) ), WithinAbs( 6.f, eps ) );
	CHECK_THAT( out( ix<1>({ 1 }) ), WithinAbs( 60.f, eps ) );
}

TEST_CASE( "reduceSumToDim: single-element slices", "[tensor_n][reduce_sum]" )
{
	TensorN<4> t( ix<4>({ 3, 1, 1, 1 }), 5.f );

	TensorN<1> out;
	t.reduceSumToDim( 0, out );

	REQUIRE( out.shape()[0] == 3 );
	for ( std::size_t i = 0; i < 3; ++i )
		CHECK_THAT( out( ix<1>({ i }) ), WithinAbs( 5.f, eps ) );
}

// --------------------------------------------------------------------------
// Conv backward: simplest case (1x1x3x3 input, 1x1x3x3 kernel, 1x1x1x1 out)
//
// With a 3x3 kernel on a 3x3 input (padding = 1), there is a single output
// element = dot(input, kernel).  dL/dInput = kernel, dL/dKernel = input.
// --------------------------------------------------------------------------

TEST_CASE( "conv backward: 3x3 kernel on 3x3 input yields exact gradients",
           "[tensor_n][conv_backward]" )
{
	std::vector<float> in_data{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	std::vector<float> k_data{ 9, 8, 7, 6, 5, 4, 3, 2, 1 };
	auto const in_shape = ix<4>({ 1, 1, 3, 3 });
	auto const k_shape  = ix<4>({ 1, 1, 3, 3 });

	TensorN<4> input( in_data, in_shape );
	TensorN<4> kernel( k_data, k_shape );
	TensorN<2> im2col;
	TensorN<4> output;
	TensorN<4> gemm_cnhw;
	input.im2ColConvolution( kernel, im2col, output, gemm_cnhw );

	REQUIRE( output.shape() == ix<4>({ 1, 1, 1, 1 }) );

	auto [gw, gb, go] = run_backward( kernel, im2col, in_shape, output.shape() );

	SECTION( "grad_output equals kernel values" )
	{
		REQUIRE( go.shape() == in_shape );
		for ( std::size_t i = 0; i < 9; ++i )
			CHECK_THAT( go.data()[i], WithinAbs( k_data[i], eps ) );
	}

	SECTION( "grad_weights equals input values" )
	{
		REQUIRE( gw.shape() == k_shape );
		for ( std::size_t i = 0; i < 9; ++i )
			CHECK_THAT( gw.data()[i], WithinAbs( in_data[i], eps ) );
	}

	SECTION( "grad_bias equals 1 (single output element, gradient = 1)" )
	{
		REQUIRE( gb.shape()[0] == 1 );
		CHECK_THAT( gb( ix<1>({ 0 }) ), WithinAbs( 1.f, eps ) );
	}
}

// --------------------------------------------------------------------------
// Conv backward: 1 batch, 1 channel, 4x4 input, 3x3 kernel → 2x2 output
//
// grad_output (dL/dInput) is the full convolution of the upstream gradient
// with the flipped kernel.  We verify a few positions by hand.
//
//   Kernel:  1 2 3      Input gradient at (0,0): appears only in output (0,0)
//            4 5 6      via kernel(0,0) = 1.  So dL/dInput(0,0) = 1.
//            7 8 9
//
//   Input gradient at (0,1): appears in out(0,0) via k(0,1)=2
//                            and in out(0,1) via k(0,0)=1.  Total = 3.
//
//   Input gradient at (1,1): appears in out(0,0) via k(1,1)=5,
//                            out(0,1) via k(1,0)=4,
//                            out(1,0) via k(0,1)=2,
//                            out(1,1) via k(0,0)=1.  Total = 12.
// --------------------------------------------------------------------------

TEST_CASE( "conv backward input gradient: 1x1x4x4 input, 3x3 kernel",
           "[tensor_n][conv_backward][grad_input]" )
{
	std::vector<float> in_data( 16 );
	std::iota( in_data.begin(), in_data.end(), 1.f );
	auto const in_shape = ix<4>({ 1, 1, 4, 4 });

	std::vector<float> k_data{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	auto const k_shape = ix<4>({ 1, 1, 3, 3 });

	TensorN<4> input( in_data, in_shape );
	TensorN<4> kernel( k_data, k_shape );
	TensorN<2> im2col;
	TensorN<4> output;
	TensorN<4> gemm_cnhw;
	input.im2ColConvolution( kernel, im2col, output, gemm_cnhw );

	REQUIRE( output.shape() == ix<4>({ 1, 1, 2, 2 }) );

	auto [gw, gb, go] = run_backward( kernel, im2col, in_shape, output.shape() );

	REQUIRE( go.shape() == in_shape );

	// Hand-computed expected gradient (all-ones upstream gradient).
	// Each input pixel accumulates the kernel weight for every output position
	// it contributes to.
	// clang-format off
	float expected[16] = {
		 1,  3,  5,  3,
		 5, 12, 16,  9,
		11, 24, 28, 15,
		 7, 15, 17,  9
	};
	// clang-format on
	for ( std::size_t i = 0; i < 16; ++i ) {
		CAPTURE( i );
		CHECK_THAT( go.data()[i], WithinAbs( expected[i], eps ) );
	}
}

TEST_CASE( "conv backward weight gradient: 1x1x4x4 input, 3x3 kernel",
           "[tensor_n][conv_backward][grad_weights]" )
{
	std::vector<float> in_data( 16 );
	std::iota( in_data.begin(), in_data.end(), 1.f );
	auto const in_shape = ix<4>({ 1, 1, 4, 4 });

	std::vector<float> k_data{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	auto const k_shape = ix<4>({ 1, 1, 3, 3 });

	TensorN<4> input( in_data, in_shape );
	TensorN<4> kernel( k_data, k_shape );
	TensorN<2> im2col;
	TensorN<4> output;
	TensorN<4> gemm_cnhw;
	input.im2ColConvolution( kernel, im2col, output, gemm_cnhw );

	auto [gw, gb, go] = run_backward( kernel, im2col, in_shape, output.shape() );

	REQUIRE( gw.shape() == k_shape );

	// dL/dKernel[kh][kw] = sum over valid output positions of
	//   input[h_out + kh][w_out + kw] * upstream_grad[h_out][w_out]
	// With upstream_grad = all ones: just sum the input patches.
	// Output positions are (0,0), (0,1), (1,0), (1,1) mapping to
	// top-left corners (0,0), (0,1), (1,0), (1,1) of the 3x3 patch.
	//
	// Expected grad_weights[kh][kw] = sum_{ho,wo} input[ho+kh][wo+kw]
	// where ho in {0,1}, wo in {0,1}.
	// clang-format off
	float expected[9] = {
		1+2+5+6,     2+3+6+7,     3+4+7+8,
		5+6+9+10,    6+7+10+11,   7+8+11+12,
		9+10+13+14,  10+11+14+15, 11+12+15+16
	};
	// clang-format on
	for ( std::size_t i = 0; i < 9; ++i ) {
		CAPTURE( i );
		CHECK_THAT( gw.data()[i], WithinAbs( expected[i], eps ) );
	}
}

// --------------------------------------------------------------------------
// Conv backward – bias gradient
// --------------------------------------------------------------------------

TEST_CASE( "conv backward bias gradient equals sum of incoming gradient per output channel",
           "[tensor_n][conv_backward][grad_bias]" )
{
	std::vector<float> in_data( 16 );
	std::iota( in_data.begin(), in_data.end(), 1.f );
	TensorN<4> input( in_data, ix<4>({ 1, 1, 4, 4 }) );

	std::vector<float> k_data( 9, 1.f );
	TensorN<4> kernel( k_data, ix<4>({ 1, 1, 3, 3 }) );

	TensorN<2> im2col;
	TensorN<4> output;
	TensorN<4> gemm_cnhw;
	input.im2ColConvolution( kernel, im2col, output, gemm_cnhw );

	auto [gw, gb, go] = run_backward( kernel, im2col, ix<4>({ 1, 1, 4, 4 }), output.shape() );

	REQUIRE( gb.shape()[0] == 1 );
	CHECK_THAT( gb( ix<1>({ 0 }) ), WithinAbs( 4.f, eps ) );
}

// --------------------------------------------------------------------------
// Conv backward – 2 output channels (2x1x3x3 kernel on 1x1x4x4 input)
//
// With C_out=2, the bias gradient has 2 elements, each the number of output
// spatial positions (= 4 for 2x2 output) because the upstream gradient is 1.
// --------------------------------------------------------------------------

TEST_CASE( "conv backward: 2 output channels, 1 input channel",
           "[tensor_n][conv_backward][multi_channel]" )
{
	std::vector<float> in_data( 16 );
	std::iota( in_data.begin(), in_data.end(), 1.f );
	auto const in_shape = ix<4>({ 1, 1, 4, 4 });

	std::vector<float> k_data( 18 );
	for ( std::size_t i = 0; i < 9; ++i ) k_data[i]     = static_cast<float>( i + 1 );
	for ( std::size_t i = 0; i < 9; ++i ) k_data[9 + i]  = static_cast<float>( 9 - i );
	auto const k_shape = ix<4>({ 2, 1, 3, 3 });

	TensorN<4> input( in_data, in_shape );
	TensorN<4> kernel( k_data, k_shape );
	TensorN<2> im2col;
	TensorN<4> output;
	TensorN<4> gemm_cnhw;
	input.im2ColConvolution( kernel, im2col, output, gemm_cnhw );

	REQUIRE( output.shape() == ix<4>({ 1, 2, 2, 2 }) );

	auto [gw, gb, go] = run_backward( kernel, im2col, in_shape, output.shape() );

	SECTION( "bias gradient: 2 channels, each summing 4 spatial positions" )
	{
		REQUIRE( gb.shape()[0] == 2 );
		CHECK_THAT( gb( ix<1>({ 0 }) ), WithinAbs( 4.f, eps ) );
		CHECK_THAT( gb( ix<1>({ 1 }) ), WithinAbs( 4.f, eps ) );
	}

	SECTION( "input gradient accumulates over both output channels" )
	{
		REQUIRE( go.shape() == in_shape );
		// Corner (0,0): kernel_0[0,0] + kernel_1[0,0] = 1 + 9 = 10
		CHECK_THAT( go( ix<4>({ 0, 0, 0, 0 }) ), WithinAbs( 10.f, eps ) );
		// Position (1,1) appears in all 4 output positions (2x2 output):
		//   out(0,0) via k[1][1], out(0,1) via k[1][0],
		//   out(1,0) via k[0][1], out(1,1) via k[0][0]
		// Channel 0: 5+4+2+1=12, Channel 1: 5+6+8+9=28 → 40
		CHECK_THAT( go( ix<4>({ 0, 0, 1, 1 }) ), WithinAbs( 40.f, eps ) );
	}

	SECTION( "weight gradient shape" )
	{
		REQUIRE( gw.shape() == k_shape );
	}
}

// --------------------------------------------------------------------------
// Conv backward – 2 batches
// --------------------------------------------------------------------------

TEST_CASE( "conv backward: 2 batches, 1 channel, 3x3 kernel",
           "[tensor_n][conv_backward][multi_batch]" )
{
	auto const in_shape = ix<4>({ 2, 1, 3, 3 });
	auto const k_shape  = ix<4>({ 1, 1, 3, 3 });

	std::vector<float> in_data( 18 );
	for ( std::size_t i = 0; i < 9; ++i )  in_data[i]     = 1.f;
	for ( std::size_t i = 0; i < 9; ++i )  in_data[9 + i] = 2.f;

	std::vector<float> k_data( 9, 1.f );

	TensorN<4> input( in_data, in_shape );
	TensorN<4> kernel( k_data, k_shape );
	TensorN<2> im2col;
	TensorN<4> output;
	TensorN<4> gemm_cnhw;
	input.im2ColConvolution( kernel, im2col, output, gemm_cnhw );

	REQUIRE( output.shape() == ix<4>({ 2, 1, 1, 1 }) );

	auto [gw, gb, go] = run_backward( kernel, im2col, in_shape, output.shape() );

	SECTION( "input gradient: all-ones kernel → each input element gets 1" )
	{
		REQUIRE( go.shape() == in_shape );
		for ( std::size_t i = 0; i < go.size(); ++i ) {
			CAPTURE( i );
			CHECK_THAT( go.data()[i], WithinAbs( 1.f, eps ) );
		}
	}

	SECTION( "weight gradient: sum of both batch patches" )
	{
		REQUIRE( gw.shape() == k_shape );
		// Each kernel position: batch0 contributes 1, batch1 contributes 2 → 3
		for ( std::size_t i = 0; i < 9; ++i ) {
			CAPTURE( i );
			CHECK_THAT( gw.data()[i], WithinAbs( 3.f, eps ) );
		}
	}

	SECTION( "bias gradient: 2 spatial positions (one per batch) → sum = 2" )
	{
		CHECK_THAT( gb( ix<1>({ 0 }) ), WithinAbs( 2.f, eps ) );
	}
}
