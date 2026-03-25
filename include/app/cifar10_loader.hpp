#ifndef CIFAR10_LOADER_HPP
#define CIFAR10_LOADER_HPP

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace neural {

/// CIFAR-10: 32×32 RGB, flattened as 1024 R + 1024 G + 1024 B (channel-major), values /255.
inline constexpr std::size_t cifar10_input_dim = 32 * 32 * 3;
inline constexpr int cifar10_num_classes = 10;

/// `TensorT` is typically `Tensor<float>` or `CudaTensor<float>` (must have `value_type` and
/// `(rows,cols,iter,iter)` construction).
template <typename TensorT>
struct CIFAR10Data
{
	std::vector<TensorT> images; // each (1, 3072)
	std::vector<TensorT> labels; // each (1, 10) one-hot
	std::size_t count{ 0 };
};

template <typename TensorT>
inline void append_one_cifar_record( std::uint8_t label_byte,
                                   std::uint8_t const *pixel_bytes,
                                   CIFAR10Data<TensorT> &out )
{
	std::vector<typename TensorT::value_type> flat( cifar10_input_dim );
	for ( std::size_t j = 0; j < cifar10_input_dim; ++j ) {
		float const n = static_cast<float>( pixel_bytes[j] ) / 255.0f;
		flat[j] = static_cast<typename TensorT::value_type>( n );
	}

	out.images.emplace_back( 1, cifar10_input_dim, flat.begin(), flat.end() );

	std::vector<typename TensorT::value_type> one_hot( static_cast<std::size_t>( cifar10_num_classes ),
	                              static_cast<typename TensorT::value_type>( 0.0f ) );
	one_hot[static_cast<std::size_t>( label_byte )] = static_cast<typename TensorT::value_type>( 1.0f );
	out.labels.emplace_back( 1, static_cast<std::size_t>( cifar10_num_classes ),
	                         one_hot.begin(), one_hot.end() );
	++out.count;
}

/// Read one `data_batch_*` or `test_batch` file (10_000 records of 3073 bytes).
template <typename TensorT>
inline bool append_cifar_batch_file( std::filesystem::path const &batch_path,
                                     CIFAR10Data<TensorT> &out, std::size_t max_samples_total )
{
	std::ifstream in( batch_path, std::ios::binary );
	if ( !in )
		return false;

	constexpr std::size_t kRecord = 1 + cifar10_input_dim;
	std::vector<std::uint8_t> buf( kRecord );

	for ( std::size_t i = 0; i < 10000; ++i ) {
		if ( max_samples_total > 0 && out.count >= max_samples_total )
			break;
		in.read( reinterpret_cast<char *>( buf.data() ),
		         static_cast<std::streamsize>( kRecord ) );
		if ( !in )
			break;
		append_one_cifar_record( buf[0], buf.data() + 1, out );
	}
	return true;
}

template <typename TensorT>
inline CIFAR10Data<TensorT> load_cifar10_train( std::filesystem::path const &bin_dir,
                                       std::size_t max_samples = 0 )
{
	CIFAR10Data<TensorT> result;
	static char const *const kBatches[] = { "data_batch_1.bin", "data_batch_2.bin",
		                                "data_batch_3.bin", "data_batch_4.bin",
		                                "data_batch_5.bin" };
	for ( char const *name : kBatches ) {
		if ( max_samples > 0 && result.count >= max_samples )
			break;
		std::filesystem::path const p = bin_dir / name;
		if ( !std::filesystem::exists( p ) )
			continue;
		append_cifar_batch_file( p, result, max_samples );
	}
	result.count = result.images.size();
	return result;
}

template <typename TensorT>
inline CIFAR10Data<TensorT> load_cifar10_test( std::filesystem::path const &bin_dir,
                                      std::size_t max_samples = 0 )
{
	CIFAR10Data<TensorT> result;
	std::filesystem::path const p = bin_dir / "test_batch.bin";
	if ( std::filesystem::exists( p ) )
		append_cifar_batch_file( p, result, max_samples );
	result.count = result.images.size();
	return result;
}

/// Looks for `data_batch_1.bin` under known layout paths.
inline std::filesystem::path find_cifar10_bin_dir()
{
	static char const *const kCandidates[] = {
		"data/cifar/cifar-10-batches-bin",
		"../data/cifar/cifar-10-batches-bin",
		"data/cifar-10-batches-bin",
		"../data/cifar-10-batches-bin",
	};
	for ( char const *c : kCandidates ) {
		std::filesystem::path const p = c;
		if ( std::filesystem::exists( p / "data_batch_1.bin" ) )
			return p;
	}
	return {};
}

} // namespace neural

#endif
