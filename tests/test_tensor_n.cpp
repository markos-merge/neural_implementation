#include "tensor_n.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <algorithm>
#include <array>
#include <vector>
#include <stdexcept>

using neural::TensorN;
using Catch::Matchers::WithinAbs;

template <std::size_t N>
std::array<std::size_t, N> idx( std::initializer_list<std::size_t> il )
{
    std::array<std::size_t, N> a{};
    std::size_t i = 0;
    for ( auto v : il ) a[i++] = v;
    return a;
}

TEST_CASE( "assign(slice) copies a full 2-D source into a matching-size dest", "[tensor_n][assign_slice]" )
{
    std::vector<float> const src_data{ 1.f, 2.f, 3.f, 4.f };
    TensorN<2> const src( src_data, idx<2>({2, 2}) );

    TensorN<2> dest( idx<2>({2, 2}), 0.f );
    dest.assign( idx<2>({0, 0}), idx<2>({2, 2}), src );

    CHECK_THAT( dest( idx<2>({0,0}) ), WithinAbs( 1.f, 1e-6f ) );
    CHECK_THAT( dest( idx<2>({0,1}) ), WithinAbs( 2.f, 1e-6f ) );
    CHECK_THAT( dest( idx<2>({1,0}) ), WithinAbs( 3.f, 1e-6f ) );
    CHECK_THAT( dest( idx<2>({1,1}) ), WithinAbs( 4.f, 1e-6f ) );
}

TEST_CASE( "assign(slice) copies into a sub-region, leaving surroundings untouched", "[tensor_n][assign_slice]" )
{
    std::vector<float> const src_data{ 10.f, 20.f, 30.f, 40.f };
    TensorN<2> const src( src_data, idx<2>({2, 2}) );

    TensorN<2> dest( idx<2>({3, 3}), 0.f );
    dest.assign( idx<2>({1, 1}), idx<2>({3, 3}), src );

    CHECK_THAT( dest( idx<2>({1,1}) ), WithinAbs( 10.f, 1e-6f ) );
    CHECK_THAT( dest( idx<2>({1,2}) ), WithinAbs( 20.f, 1e-6f ) );
    CHECK_THAT( dest( idx<2>({2,1}) ), WithinAbs( 30.f, 1e-6f ) );
    CHECK_THAT( dest( idx<2>({2,2}) ), WithinAbs( 40.f, 1e-6f ) );

    CHECK_THAT( dest( idx<2>({0,0}) ), WithinAbs( 0.f, 1e-6f ) );
    CHECK_THAT( dest( idx<2>({0,2}) ), WithinAbs( 0.f, 1e-6f ) );
    CHECK_THAT( dest( idx<2>({2,0}) ), WithinAbs( 0.f, 1e-6f ) );
}

TEST_CASE( "assign(slice) overwrites previously written values", "[tensor_n][assign_slice]" )
{
    TensorN<2> dest( idx<2>({2, 2}), 9.f );
    TensorN<2> const src( std::vector<float>{1.f, 2.f, 3.f, 4.f}, idx<2>({2, 2}) );

    dest.assign( idx<2>({0, 0}), idx<2>({2, 2}), src );

    CHECK_THAT( dest( idx<2>({0, 0}) ), WithinAbs( 1.f, 1e-6f ) );
    CHECK_THAT( dest( idx<2>({1, 1}) ), WithinAbs( 4.f, 1e-6f ) );
}

TEST_CASE( "assign(slice) works for a single-element region", "[tensor_n][assign_slice]" )
{
    TensorN<2> dest( idx<2>({3, 3}), 0.f );
    TensorN<2> const src( std::vector<float>{7.f}, idx<2>({1, 1}) );

    dest.assign( idx<2>({2, 2}), idx<2>({3, 3}), src );

    CHECK_THAT( dest( idx<2>({2, 2}) ), WithinAbs( 7.f, 1e-6f ) );
    CHECK_THAT( dest( idx<2>({0, 0}) ), WithinAbs( 0.f, 1e-6f ) );
}

TEST_CASE( "assign(slice) works for a 1-D tensor", "[tensor_n][assign_slice]" )
{
    TensorN<1> dest( idx<1>({5}), 0.f );
    TensorN<1> const src( std::vector<float>{10.f, 20.f, 30.f}, idx<1>({3}) );

    dest.assign( idx<1>({1}), idx<1>({4}), src );

    CHECK_THAT( dest( idx<1>({0}) ), WithinAbs(  0.f, 1e-6f ) );
    CHECK_THAT( dest( idx<1>({1}) ), WithinAbs( 10.f, 1e-6f ) );
    CHECK_THAT( dest( idx<1>({2}) ), WithinAbs( 20.f, 1e-6f ) );
    CHECK_THAT( dest( idx<1>({3}) ), WithinAbs( 30.f, 1e-6f ) );
    CHECK_THAT( dest( idx<1>({4}) ), WithinAbs(  0.f, 1e-6f ) );
}

TEST_CASE( "assign(slice) works for a 3-D tensor", "[tensor_n][assign_slice]" )
{
    std::vector<float> const src_data{ 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };
    TensorN<3> const src( src_data, idx<3>({2, 2, 2}) );

    TensorN<3> dest( idx<3>({3, 3, 3}), 0.f );
    dest.assign( idx<3>({1, 1, 1}), idx<3>({3, 3, 3}), src );

    for ( std::size_t i = 1; i < 3; ++i )
        for ( std::size_t j = 1; j < 3; ++j )
            for ( std::size_t k = 1; k < 3; ++k )
                CHECK( dest( idx<3>({i, j, k}) ) != 0.f );

    CHECK_THAT( dest( idx<3>({0,0,0}) ), WithinAbs( 0.f, 1e-6f ) );
}

TEST_CASE( "assign(slice) throws when region element count differs from source size", "[tensor_n][assign_slice]" )
{
    TensorN<2> dest( idx<2>({4, 4}), 0.f );
    TensorN<2> const src( idx<2>({2, 2}), 1.f );

    REQUIRE_NOTHROW( dest.assign( idx<2>({1, 1}), idx<2>({3, 3}), src ) );
    REQUIRE_THROWS_AS( dest.assign( idx<2>({0, 0}), idx<2>({3, 3}), src ), std::invalid_argument );
}

TEST_CASE( "assign(slice) throws when any axis has an empty or inverted range", "[tensor_n][assign_slice]" )
{
    TensorN<2> dest( idx<2>({3, 3}), 0.f );
    TensorN<2> const src( idx<2>({1, 1}), 1.f );

    REQUIRE_THROWS_AS( dest.assign( idx<2>({2, 0}), idx<2>({2, 1}), src ), std::invalid_argument );
    REQUIRE_THROWS_AS( dest.assign( idx<2>({0, 2}), idx<2>({1, 1}), src ), std::invalid_argument );
}

// im2Col: NCHW row-major input, kernel_shape = {C_k, K_h, K_w}. Expected layout is documented in
// TENSOR_N.md; these expectations match standard im2col (one column per output spatial cell per batch).

TEST_CASE( "im2Col 1x1x3x3 with 3x3 kernel yields 9x1 matrix with raster patch order", "[tensor_n][im2col]" )
{
    // Single batch, one channel: 3x3 plane with values 1..9 in row-major (W fastest).
    std::vector<float> const data{ 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f };
    TensorN<4> const input( data, idx<4>({1, 1, 3, 3}) );

    TensorN<2> out;
    input.im2Col( idx<3>({1, 3, 3}), out );

    REQUIRE( out.shape()[0] == 9 );
    REQUIRE( out.shape()[1] == 1 );

    for ( std::size_t r = 0; r < 9; ++r )
        CHECK_THAT( out( idx<2>({r, 0}) ), WithinAbs( static_cast<float>( r + 1 ), 1e-6f ) );
}

TEST_CASE( "im2Col 2x1x3x3 with 3x3 kernel separates batches into adjacent columns", "[tensor_n][im2col]" )
{
    std::vector<float> data( 18 );
    for ( std::size_t i = 0; i < 9; ++i )
        data[i] = 1.f;
    for ( std::size_t i = 9; i < 18; ++i )
        data[i] = 2.f;

    TensorN<4> const input( data, idx<4>({2, 1, 3, 3}) );
    TensorN<2> out;
    input.im2Col( idx<3>({1, 3, 3}), out );

    REQUIRE( out.shape()[0] == 9 );
    REQUIRE( out.shape()[1] == 2 );

    for ( std::size_t r = 0; r < 9; ++r ) {
        CHECK_THAT( out( idx<2>({r, 0}) ), WithinAbs( 1.f, 1e-6f ) );
        CHECK_THAT( out( idx<2>({r, 1}) ), WithinAbs( 2.f, 1e-6f ) );
    }
}

TEST_CASE( "im2ColConvolution 1x1x3x3 with 2x2 all-ones kernel (padding K/2) one output cell", "[tensor_n][im2col_conv]" )
{
    // im2Col uses padding_i = Kh/2, padding_j = Kw/2 → H_out = H - 2*padding = 1 for H=3, Kh=2.
    std::vector<float> const in_data{ 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f };
    TensorN<4> input( in_data, idx<4>({1, 1, 3, 3}) );

    std::vector<float> const k_data{ 1.f, 1.f, 1.f, 1.f };
    TensorN<4> const kernel( k_data, idx<4>({1, 1, 2, 2}) );

    TensorN<2> im2col;
    TensorN<4> output;
    TensorN<4> gemm_cnhw;
    input.im2ColConvolution( kernel, im2col, output, gemm_cnhw );

    REQUIRE( output.shape() == idx<4>({1, 1, 1, 1}) );
    // Only valid window is the top-left 2x2 block {1,2,4,5} → sum 12.
    CHECK_THAT( output( idx<4>({0, 0, 0, 0}) ), WithinAbs( 12.f, 1e-4f ) );
}

TEST_CASE( "im2ColConvolution 1x1x3x3 with 3x3 all-ones kernel sums entire plane to 45", "[tensor_n][im2col_conv]" )
{
    std::vector<float> const in_data{ 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f };
    TensorN<4> input( in_data, idx<4>({1, 1, 3, 3}) );

    std::vector<float> const k_data( 9, 1.f );
    TensorN<4> const kernel( k_data, idx<4>({1, 1, 3, 3}) );

    TensorN<2> im2col;
    TensorN<4> output;
    TensorN<4> gemm_cnhw;
    input.im2ColConvolution( kernel, im2col, output, gemm_cnhw );

    REQUIRE( output.shape() == idx<4>({1, 1, 1, 1}) );
    CHECK_THAT( output( idx<4>({0, 0, 0, 0}) ), WithinAbs( 45.f, 1e-4f ) );
}

TEST_CASE( "im2ColConvolution two batches 3x3 with 3x3 all-ones kernel sums each batch plane", "[tensor_n][im2col_conv]" )
{
    std::vector<float> in_data( 18 );
    for ( std::size_t i = 0; i < 9; ++i )
        in_data[i] = 1.f;
    for ( std::size_t i = 9; i < 18; ++i )
        in_data[i] = 2.f;

    TensorN<4> input( in_data, idx<4>({2, 1, 3, 3}) );
    std::vector<float> const k_data( 9, 1.f );
    TensorN<4> const kernel( k_data, idx<4>({1, 1, 3, 3}) );

    TensorN<2> im2col;
    TensorN<4> output;
    TensorN<4> gemm_cnhw;
    input.im2ColConvolution( kernel, im2col, output, gemm_cnhw );

    REQUIRE( output.shape() == idx<4>({2, 1, 1, 1}) );
    CHECK_THAT( output( idx<4>({0, 0, 0, 0}) ), WithinAbs( 9.f, 1e-4f ) );
    CHECK_THAT( output( idx<4>({1, 0, 0, 0}) ), WithinAbs( 18.f, 1e-4f ) );
}

TEST_CASE( "im2Col 1x2x3x3 with 3x3x2 kernel stacks channels in row-major kernel order", "[tensor_n][im2col]" )
{
    // Two channels: ch0 = 1..9, ch1 = 10..18 (row-major each).
    std::vector<float> data( 18 );
    for ( std::size_t i = 0; i < 9; ++i )
        data[i] = static_cast<float>( i + 1 );
    for ( std::size_t i = 0; i < 9; ++i )
        data[9 + i] = static_cast<float>( 10 + i );

    TensorN<4> const input( data, idx<4>({1, 2, 3, 3}) );
    TensorN<2> out;
    input.im2Col( idx<3>({2, 3, 3}), out );

    REQUIRE( out.shape()[0] == 18 );
    REQUIRE( out.shape()[1] == 1 );

    for ( std::size_t r = 0; r < 9; ++r )
        CHECK_THAT( out( idx<2>({r, 0}) ), WithinAbs( static_cast<float>( r + 1 ), 1e-6f ) );
    for ( std::size_t r = 0; r < 9; ++r )
        CHECK_THAT( out( idx<2>({9 + r, 0}) ), WithinAbs( static_cast<float>( 10 + r ), 1e-6f ) );
}

namespace {

// col2Im is on TensorN<4>; im2col storage must use shape { rows, cols, 1, 1 } so column stride matches im2Col().
TensorN<4> im2colAsRank4( TensorN<2> const &m )
{
    std::vector<float> buf( m.size() );
    std::copy( m.data(), m.data() + m.size(), buf.begin() );
    return TensorN<4>( buf, idx<4>({ m.shape()[0], m.shape()[1], 1, 1 }) );
}

} // namespace

TEST_CASE( "col2Im round-trip 1x1x3x3 with 3x3 kernel recovers input", "[tensor_n][col2im]" )
{
    std::vector<float> const data{ 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f };
    TensorN<4> const input( data, idx<4>({1, 1, 3, 3}) );
    TensorN<2> m2;
    input.im2Col( idx<3>({1, 3, 3}), m2 );

    TensorN<4> m4 = im2colAsRank4( m2 );
    std::array<std::size_t, 4> const original_shape = idx<4>({1, 1, 3, 3});
    TensorN<4> recovered( original_shape, 0.f );
    m4.col2Im( {1, 1, 3, 3}, original_shape, recovered );

    for ( std::size_t i = 0; i < 9; ++i )
        CHECK_THAT( recovered.data()[i], WithinAbs( data[i], 1e-5f ) );
}

TEST_CASE( "col2Im round-trip 2x1x3x3 with 3x3 kernel keeps batch columns separate", "[tensor_n][col2im]" )
{
    std::vector<float> data( 18 );
    for ( std::size_t i = 0; i < 9; ++i )
        data[i] = 1.f;
    for ( std::size_t i = 9; i < 18; ++i )
        data[i] = 2.f;

    TensorN<4> const input( data, idx<4>({2, 1, 3, 3}) );
    TensorN<2> m2;
    input.im2Col( idx<3>({1, 3, 3}), m2 );

    TensorN<4> m4 = im2colAsRank4( m2 );
    std::array<std::size_t, 4> const original_shape = idx<4>({2, 1, 3, 3});
    TensorN<4> recovered( original_shape, 0.f );
    m4.col2Im( {1, 1, 3, 3}, original_shape, recovered );

    for ( std::size_t i = 0; i < 9; ++i )
        CHECK_THAT( recovered.data()[i], WithinAbs( 1.f, 1e-5f ) );
    for ( std::size_t i = 9; i < 18; ++i )
        CHECK_THAT( recovered.data()[i], WithinAbs( 2.f, 1e-5f ) );
}

TEST_CASE( "col2Im round-trip 1x2x3x3 with two-channel 3x3 kernel", "[tensor_n][col2im]" )
{
    std::vector<float> data( 18 );
    for ( std::size_t i = 0; i < 9; ++i )
        data[i] = static_cast<float>( i + 1 );
    for ( std::size_t i = 0; i < 9; ++i )
        data[9 + i] = static_cast<float>( 10 + i );

    TensorN<4> const input( data, idx<4>({1, 2, 3, 3}) );
    TensorN<2> m2;
    input.im2Col( idx<3>({2, 3, 3}), m2 );

    TensorN<4> m4 = im2colAsRank4( m2 );
    std::array<std::size_t, 4> const original_shape = idx<4>({1, 2, 3, 3});
    TensorN<4> recovered( original_shape, 0.f );
    m4.col2Im( {1, 2, 3, 3}, original_shape, recovered );

    for ( std::size_t i = 0; i < 18; ++i )
        CHECK_THAT( recovered.data()[i], WithinAbs( data[i], 1e-5f ) );
}

TEST_CASE( "col2Im single activated row maps to one image pixel", "[tensor_n][col2im]" )
{
    TensorN<4> m4( idx<4>({9, 1, 1, 1}), 0.f );
    m4.data()[4] = 1.f;

    std::array<std::size_t, 4> const original_shape = idx<4>({1, 1, 3, 3});
    TensorN<4> out( original_shape, 0.f );
    m4.col2Im( {1, 1, 3, 3}, original_shape, out );

    CHECK_THAT( out( idx<4>({0, 0, 1, 1}) ), WithinAbs( 1.f, 1e-5f ) );
    for ( std::size_t h = 0; h < 3; ++h ) {
        for ( std::size_t w = 0; w < 3; ++w ) {
            if ( h == 1 && w == 1 )
                continue;
            CHECK_THAT( out( idx<4>({0, 0, h, w}) ), WithinAbs( 0.f, 1e-5f ) );
        }
    }
}
