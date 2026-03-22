#ifndef WEIGHT_DEBUG_HPP
#define WEIGHT_DEBUG_HPP

#include "tensor_like.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace neural {

/// Print min / max / mean / L2 norm (handy while stepping in a debugger or reading logs).
template <TensorLike Tensor>
void print_weight_stats( std::string const &name, Tensor const &t )
{
	using T = typename Tensor::value_type;
	if ( t.size() == 0 ) {
		std::cout << "[weights] " << name << ": empty\n";
		return;
	}
	T mn = t( 0, 0 );
	T mx = t( 0, 0 );
	for ( std::size_t i = 0; i < t.rows(); ++i ) {
		for ( std::size_t j = 0; j < t.cols(); ++j ) {
			T v = t( i, j );
			mn = std::min( mn, v );
			mx = std::max( mx, v );
		}
	}
	T mean = t.sum() / static_cast<T>( t.size() );
	T sq = 0;
	for ( std::size_t i = 0; i < t.rows(); ++i ) {
		for ( std::size_t j = 0; j < t.cols(); ++j ) {
			T v = t( i, j );
			sq += v * v;
		}
	}
	T l2 = static_cast<T>( std::sqrt( static_cast<double>( sq ) ) );
	std::cout << "[weights] " << name << "  shape=" << t.rows() << "x" << t.cols()
	          << "  min=" << mn << " max=" << mx << " mean=" << mean << "  L2=" << l2
	          << "\n";
}

/// Write a binary PGM (P5) grayscale image.
inline bool write_grayscale_pgm( std::filesystem::path const &path,
                                 std::size_t width, std::size_t height,
                                 std::vector<std::uint8_t> const &pixels )
{
	if ( pixels.size() != width * height )
		return false;
	std::ofstream out( path, std::ios::binary );
	if ( !out )
		return false;
	out << "P5\n" << width << " " << height << "\n255\n";
	out.write( reinterpret_cast<char const *>( pixels.data() ),
	           static_cast<std::streamsize>( pixels.size() ) );
	return static_cast<bool>( out );
}

/// Visualize the first fully-connected layer when inputs are 28×28 MNIST pixels (784 rows).
/// Each column of `W` is one neuron; weights are laid out as row-major 28×28 patches in a grid.
template <TensorLike Tensor>
bool save_mnist_first_layer_weight_grid( Tensor const &W,
                                         std::filesystem::path const &path,
                                         std::size_t grid_cols,
                                         std::size_t max_neurons )
{
	static constexpr std::size_t kSide = 28;
	static constexpr std::size_t kIn = kSide * kSide;
	if ( W.rows() != kIn ) {
		std::cerr << "save_mnist_first_layer_weight_grid: expected " << kIn
		          << " rows, got " << W.rows() << "\n";
		return false;
	}
	if ( grid_cols == 0 || max_neurons == 0 )
		return false;

	using T = typename Tensor::value_type;
	std::size_t const n_neurons = std::min( W.cols(), max_neurons );
	std::size_t const grid_rows = ( n_neurons + grid_cols - 1 ) / grid_cols;
	std::size_t const tw = grid_cols * kSide;
	std::size_t const th = grid_rows * kSide;
	std::vector<std::uint8_t> buf( tw * th, 0 );

	for ( std::size_t n = 0; n < n_neurons; ++n ) {
		T mn = W( 0, n );
		T mx = W( 0, n );
		for ( std::size_t r = 1; r < kIn; ++r ) {
			T v = W( r, n );
			mn = std::min( mn, v );
			mx = std::max( mx, v );
		}
		T const scale =
		    ( mx > mn ) ? static_cast<T>( 255 ) / ( mx - mn ) : static_cast<T>( 0 );
		std::size_t const cell_x = ( n % grid_cols ) * kSide;
		std::size_t const cell_y = ( n / grid_cols ) * kSide;
		for ( std::size_t y = 0; y < kSide; ++y ) {
			for ( std::size_t x = 0; x < kSide; ++x ) {
				std::size_t const r = y * kSide + x;
				T const v = W( r, n );
				std::uint8_t p =
				    ( mx > mn )
				        ? static_cast<std::uint8_t>(
				              std::clamp( static_cast<double>( ( v - mn ) * scale ),
				                          0.0, 255.0 ) )
				        : static_cast<std::uint8_t>( 128 );
				std::size_t const gx = cell_x + x;
				std::size_t const gy = cell_y + y;
				buf[gy * tw + gx] = p;
			}
		}
	}

	if ( path.has_parent_path() )
		std::filesystem::create_directories( path.parent_path() );
	return write_grayscale_pgm( path, tw, th, buf );
}

} // namespace neural

#endif
