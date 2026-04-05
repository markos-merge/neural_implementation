#include "max_pool_layer.hpp"
#include "tensor_n.hpp"
#include "layer_wiring_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <array>
#include <numeric>
#include <vector>

using neural::MaxPoolLayer;
using neural::TensorN;
using neural::test::wire_layer;
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
} // namespace

// --------------------------------------------------------------------------
// Forward: output shape
// --------------------------------------------------------------------------

TEST_CASE( "MaxPoolLayer forward output shape: 4x4 input, 2x2 pool, stride 2",
           "[max_pool][forward][shape]" )
{
	MaxPoolLayer<TensorN<4>> layer( 2, 2 );

	TensorN<4> in_buf( ix<4>({ 1, 1, 4, 4 }) );
	TensorN<4> out_buf( ix<4>({ 1, 1, 1, 1 }) );
	TensorN<4> g_out( ix<4>({ 1, 1, 1, 1 }) );
	TensorN<4> g_in( ix<4>({ 1, 1, 4, 4 }) );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );

	layer.forward();

	REQUIRE( out_buf.shape() == ix<4>({ 1, 1, 2, 2 }) );
}

TEST_CASE( "MaxPoolLayer forward output shape: 5x5 input, 2x2 pool, stride 2 (floor drops remainder)",
           "[max_pool][forward][shape]" )
{
	MaxPoolLayer<TensorN<4>> layer( 2, 2 );

	TensorN<4> in_buf( ix<4>({ 1, 1, 5, 5 }) );
	TensorN<4> out_buf( ix<4>({ 1, 1, 1, 1 }) );
	TensorN<4> g_out( ix<4>({ 1, 1, 1, 1 }) );
	TensorN<4> g_in( ix<4>({ 1, 1, 5, 5 }) );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );

	layer.forward();

	// floor((5-2)/2) + 1 = 2
	REQUIRE( out_buf.shape() == ix<4>({ 1, 1, 2, 2 }) );
}

// --------------------------------------------------------------------------
// Forward: correct max values, non-overlapping (stride == pool_size)
//
//  Input (1x1x4x4):
//   1  2  3  4
//   5  6  7  8
//   9 10 11 12
//  13 14 15 16
//
//  2x2 pool, stride 2 — four non-overlapping windows:
//  top-left:     max(1,2,5,6)    = 6
//  top-right:    max(3,4,7,8)    = 8
//  bottom-left:  max(9,10,13,14) = 14
//  bottom-right: max(11,12,15,16)= 16
// --------------------------------------------------------------------------

TEST_CASE( "MaxPoolLayer forward values: 2x2 pool stride 2 on 4x4 input",
           "[max_pool][forward][values]" )
{
	MaxPoolLayer<TensorN<4>> layer( 2, 2 );

	std::vector<float> in_data( 16 );
	std::iota( in_data.begin(), in_data.end(), 1.f );
	TensorN<4> in_buf( in_data, ix<4>({ 1, 1, 4, 4 }) );
	TensorN<4> out_buf( ix<4>({ 1, 1, 1, 1 }) );
	TensorN<4> g_out( ix<4>({ 1, 1, 1, 1 }) );
	TensorN<4> g_in( ix<4>({ 1, 1, 4, 4 }) );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );

	layer.forward();

	CHECK_THAT( out_buf( ix<4>({ 0, 0, 0, 0 }) ), WithinAbs(  6.f, eps ) );
	CHECK_THAT( out_buf( ix<4>({ 0, 0, 0, 1 }) ), WithinAbs(  8.f, eps ) );
	CHECK_THAT( out_buf( ix<4>({ 0, 0, 1, 0 }) ), WithinAbs( 14.f, eps ) );
	CHECK_THAT( out_buf( ix<4>({ 0, 0, 1, 1 }) ), WithinAbs( 16.f, eps ) );
}

// --------------------------------------------------------------------------
// Forward: correct max values, overlapping (stride 1)
//
//  Input (1x1x3x3):
//  1 2 3
//  4 5 6
//  7 8 9
//
//  2x2 pool, stride 1 — output shape 2x2:
//  (0,0): max(1,2,4,5) = 5
//  (0,1): max(2,3,5,6) = 6
//  (1,0): max(4,5,7,8) = 8
//  (1,1): max(5,6,8,9) = 9
// --------------------------------------------------------------------------

TEST_CASE( "MaxPoolLayer forward values: 2x2 pool stride 1 on 3x3 input",
           "[max_pool][forward][values][overlapping]" )
{
	MaxPoolLayer<TensorN<4>> layer( 2, 1 );

	std::vector<float> in_data( 9 );
	std::iota( in_data.begin(), in_data.end(), 1.f );
	TensorN<4> in_buf( in_data, ix<4>({ 1, 1, 3, 3 }) );
	TensorN<4> out_buf( ix<4>({ 1, 1, 1, 1 }) );
	TensorN<4> g_out( ix<4>({ 1, 1, 1, 1 }) );
	TensorN<4> g_in( ix<4>({ 1, 1, 3, 3 }) );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );

	layer.forward();

	REQUIRE( out_buf.shape() == ix<4>({ 1, 1, 2, 2 }) );
	CHECK_THAT( out_buf( ix<4>({ 0, 0, 0, 0 }) ), WithinAbs( 5.f, eps ) );
	CHECK_THAT( out_buf( ix<4>({ 0, 0, 0, 1 }) ), WithinAbs( 6.f, eps ) );
	CHECK_THAT( out_buf( ix<4>({ 0, 0, 1, 0 }) ), WithinAbs( 8.f, eps ) );
	CHECK_THAT( out_buf( ix<4>({ 0, 0, 1, 1 }) ), WithinAbs( 9.f, eps ) );
}

// --------------------------------------------------------------------------
// Forward: multiple channels are pooled independently
// --------------------------------------------------------------------------

TEST_CASE( "MaxPoolLayer forward: two channels pooled independently",
           "[max_pool][forward][multi_channel]" )
{
	MaxPoolLayer<TensorN<4>> layer( 2, 2 );

	// channel 0: all 1s; channel 1: all 2s
	TensorN<4> in_buf( ix<4>({ 1, 2, 4, 4 }) );
	for ( std::size_t i = 0; i < 16; ++i ) in_buf.data()[i]      = 1.f;
	for ( std::size_t i = 0; i < 16; ++i ) in_buf.data()[16 + i] = 2.f;

	TensorN<4> out_buf( ix<4>({ 1, 1, 1, 1 }) );
	TensorN<4> g_out( ix<4>({ 1, 1, 1, 1 }) );
	TensorN<4> g_in( ix<4>({ 1, 2, 4, 4 }) );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );

	layer.forward();

	REQUIRE( out_buf.shape() == ix<4>({ 1, 2, 2, 2 }) );
	for ( std::size_t oh = 0; oh < 2; ++oh )
		for ( std::size_t ow = 0; ow < 2; ++ow ) {
			CHECK_THAT( out_buf( ix<4>({ 0, 0, oh, ow }) ), WithinAbs( 1.f, eps ) );
			CHECK_THAT( out_buf( ix<4>({ 0, 1, oh, ow }) ), WithinAbs( 2.f, eps ) );
		}
}

// --------------------------------------------------------------------------
// Backward: gradient routed only to max positions (non-overlapping)
//
//  Same 4x4 iota input, 2x2 pool stride 2.
//  Maxes are at input positions (1,1), (1,3), (3,1), (3,3).
//  All upstream gradients = 1 → those four positions get 1, all others 0.
// --------------------------------------------------------------------------

TEST_CASE( "MaxPoolLayer backward: gradient at max positions only",
           "[max_pool][backward][grad_routing]" )
{
	MaxPoolLayer<TensorN<4>> layer( 2, 2 );

	std::vector<float> in_data( 16 );
	std::iota( in_data.begin(), in_data.end(), 1.f );
	TensorN<4> in_buf( in_data, ix<4>({ 1, 1, 4, 4 }) );
	TensorN<4> out_buf( ix<4>({ 1, 1, 1, 1 }) );
	TensorN<4> g_out( ix<4>({ 1, 1, 1, 1 }) );
	TensorN<4> g_in( ix<4>({ 1, 1, 4, 4 }) );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );

	layer.forward();

	// upstream gradient: all ones, sized to match the pooling output
	g_out = TensorN<4>( out_buf.shape(), 1.f );
	layer.backward();

	// max positions receive gradient 1
	CHECK_THAT( g_in( ix<4>({ 0, 0, 1, 1 }) ), WithinAbs( 1.f, eps ) );
	CHECK_THAT( g_in( ix<4>({ 0, 0, 1, 3 }) ), WithinAbs( 1.f, eps ) );
	CHECK_THAT( g_in( ix<4>({ 0, 0, 3, 1 }) ), WithinAbs( 1.f, eps ) );
	CHECK_THAT( g_in( ix<4>({ 0, 0, 3, 3 }) ), WithinAbs( 1.f, eps ) );

	// non-max positions receive gradient 0
	CHECK_THAT( g_in( ix<4>({ 0, 0, 0, 0 }) ), WithinAbs( 0.f, eps ) );
	CHECK_THAT( g_in( ix<4>({ 0, 0, 0, 1 }) ), WithinAbs( 0.f, eps ) );
	CHECK_THAT( g_in( ix<4>({ 0, 0, 1, 0 }) ), WithinAbs( 0.f, eps ) );
	CHECK_THAT( g_in( ix<4>({ 0, 0, 2, 2 }) ), WithinAbs( 0.f, eps ) );
}

// --------------------------------------------------------------------------
// Backward: gradient accumulates when windows overlap (stride < pool_size)
//
//  Input (1x1x3x2):
//  1 1
//  1 2
//  1 1
//
//  2x2 pool, stride 1 → output shape (2,1):
//  window (0,0): max(1,1,1,2) = 2 at local (1,1) → input (1,1)
//  window (1,0): max(1,2,1,1) = 2 at local (0,1) → input (1,1)
//
//  Both output positions point to input (1,1).
//  With upstream gradient = 1 → grad at input(1,1) = 2.
// --------------------------------------------------------------------------

TEST_CASE( "MaxPoolLayer backward: gradient accumulates for overlapping windows",
           "[max_pool][backward][accumulation]" )
{
	MaxPoolLayer<TensorN<4>> layer( 2, 1 );

	std::vector<float> in_data{ 1.f, 1.f, 1.f, 2.f, 1.f, 1.f };
	TensorN<4> in_buf( in_data, ix<4>({ 1, 1, 3, 2 }) );
	TensorN<4> out_buf( ix<4>({ 1, 1, 1, 1 }) );
	TensorN<4> g_out( ix<4>({ 1, 1, 1, 1 }) );
	TensorN<4> g_in( ix<4>({ 1, 1, 3, 2 }) );
	wire_layer( layer, in_buf, out_buf, g_out, g_in );

	layer.forward();

	REQUIRE( out_buf.shape() == ix<4>({ 1, 1, 2, 1 }) );

	g_out = TensorN<4>( out_buf.shape(), 1.f );
	layer.backward();

	// input(1,1) is the argmax for both output positions → receives 2
	CHECK_THAT( g_in( ix<4>({ 0, 0, 1, 1 }) ), WithinAbs( 2.f, eps ) );
	// all other positions are non-max → 0
	CHECK_THAT( g_in( ix<4>({ 0, 0, 0, 0 }) ), WithinAbs( 0.f, eps ) );
	CHECK_THAT( g_in( ix<4>({ 0, 0, 0, 1 }) ), WithinAbs( 0.f, eps ) );
	CHECK_THAT( g_in( ix<4>({ 0, 0, 1, 0 }) ), WithinAbs( 0.f, eps ) );
	CHECK_THAT( g_in( ix<4>({ 0, 0, 2, 0 }) ), WithinAbs( 0.f, eps ) );
	CHECK_THAT( g_in( ix<4>({ 0, 0, 2, 1 }) ), WithinAbs( 0.f, eps ) );
}
