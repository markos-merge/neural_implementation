#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include "tensor.hpp"
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace neural {

inline std::uint32_t read_be32( std::istream &in )
{
	std::uint8_t buf[4];
	in.read( reinterpret_cast<char *>( buf ), 4 );
	return ( static_cast<std::uint32_t>( buf[0] ) << 24 ) |
	       ( static_cast<std::uint32_t>( buf[1] ) << 16 ) |
	       ( static_cast<std::uint32_t>( buf[2] ) << 8 ) |
	       static_cast<std::uint32_t>( buf[3] );
}

struct MNISTData
{
	std::vector<Tensor<float>> images;  // each (1, 784)
	std::vector<Tensor<float>> labels;  // each (1, 10) one-hot
	std::size_t count{ 0 };
};

inline MNISTData load_mnist_impl( std::string const &data_dir,
                                  std::string const &img_name,
                                  std::string const &lbl_name,
                                  std::size_t max_samples )
{
	MNISTData result;

	std::ifstream img_file( data_dir + img_name, std::ios::binary );
	std::ifstream lbl_file( data_dir + lbl_name, std::ios::binary );
	if ( !img_file.is_open() || !lbl_file.is_open() )
		return result;

	std::uint32_t img_magic = read_be32( img_file );
	std::uint32_t img_count = read_be32( img_file );
	std::uint32_t img_rows = read_be32( img_file );
	std::uint32_t img_cols = read_be32( img_file );

	std::uint32_t lbl_magic = read_be32( lbl_file );
	std::uint32_t lbl_count = read_be32( lbl_file );

	if ( img_magic != 2051 || lbl_magic != 2049 || img_count != lbl_count ) {
		return result;
	}

	std::size_t total = static_cast<std::size_t>( img_count );
	if ( max_samples > 0 && max_samples < total )
		total = max_samples;

	std::size_t const pixels = static_cast<std::size_t>( img_rows * img_cols );
	std::vector<std::uint8_t> pixel_buf( pixels );

	result.images.reserve( total );
	result.labels.reserve( total );

	for ( std::size_t i = 0; i < total; ++i ) {
		img_file.read( reinterpret_cast<char *>( pixel_buf.data() ),
		               static_cast<std::streamsize>( pixels ) );
		if ( !img_file )
			break;

		std::vector<float> flat( pixels );
		for ( std::size_t k = 0; k < pixels; ++k )
			flat[k] = static_cast<float>( pixel_buf[k] ) / 255.0f;

		result.images.emplace_back(
		    1, static_cast<std::size_t>( pixels ), flat.begin(), flat.end() );

		std::uint8_t label;
		lbl_file.read( reinterpret_cast<char *>( &label ), 1 );
		if ( !lbl_file )
			break;

		std::vector<float> one_hot( 10, 0.0f );
		one_hot[static_cast<std::size_t>( label )] = 1.0f;
		result.labels.emplace_back( 1, 10, one_hot.begin(), one_hot.end() );
	}

	result.count = result.images.size();
	return result;
}

inline MNISTData load_mnist( std::string const &data_dir,
                             std::size_t max_samples = 0 )
{
	std::string const img_names[] = { "/train-images.idx3-ubyte",
	                                  "/train-images-idx3-ubyte" };
	std::string const lbl_names[] = { "/train-labels.idx1-ubyte",
	                                  "/train-labels-idx1-ubyte" };
	for ( std::size_t i = 0; i < 2; ++i ) {
		auto r = load_mnist_impl( data_dir, img_names[i], lbl_names[i], max_samples );
		if ( r.count > 0 )
			return r;
	}
	return MNISTData{};
}

inline MNISTData load_mnist_test( std::string const &data_dir,
                                  std::size_t max_samples = 0 )
{
	std::string const img_names[] = { "/t10k-images.idx3-ubyte",
	                                  "/t10k-images-idx3-ubyte" };
	std::string const lbl_names[] = { "/t10k-labels.idx1-ubyte",
	                                  "/t10k-labels-idx1-ubyte" };
	for ( std::size_t i = 0; i < 2; ++i ) {
		auto r = load_mnist_impl( data_dir, img_names[i], lbl_names[i], max_samples );
		if ( r.count > 0 )
			return r;
	}
	return MNISTData{};
}

} // namespace neural

#endif
