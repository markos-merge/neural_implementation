#ifndef REFACTOR_NN_IO_HPP
#define REFACTOR_NN_IO_HPP

#include "dtype.hpp"
#include "layer.hpp"
#include "sequential_nn.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <array>
#include <optional>

#if NEURAL_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace neural {
namespace refactor {

inline constexpr std::uint8_t kNnFormatVersion = 1;

enum class LayerType : std::uint8_t {
	Linear    = 0,
	ReLU      = 1,
	Sigmoid   = 2,
	Softmax   = 3,
	Dropout   = 4,
	BatchNorm = 5,
	Conv      = 6,
	MaxPool   = 7,
};

struct Header {
	char          magic[4]{};
	std::uint8_t  version = 0;
	DType         dtype   = DType::Float32;
	std::uint32_t num_layers = 0;
	std::uint32_t nn_type    = 0;
};

template <typename L>
struct LayerTag {};

namespace nn_io_detail {

inline void write_u16_le( std::ostream &os, std::uint16_t v )
{
	os.put( static_cast<char>( v & 0xFFU ) );
	os.put( static_cast<char>( ( v >> 8U ) & 0xFFU ) );
}

inline void write_u32_le( std::ostream &os, std::uint32_t v )
{
	for ( int i = 0; i < 4; ++i ) {
		os.put(
		    static_cast<char>( ( v >> ( 8U * static_cast<unsigned>( i ) ) ) & 0xFFU ) );
	}
}

inline void write_u64_le_to_stream( std::ostream &os, std::uint64_t v )
{
	for ( int i = 0; i < 8; ++i ) {
		os.put(
		    static_cast<char>( ( v >> ( 8U * static_cast<unsigned>( i ) ) ) & 0xFFU ) );
	}
}

inline std::uint16_t read_u16_le( std::istream &is )
{
	std::uint16_t v = 0;
	for ( int i = 0; i < 2; ++i ) {
		char c{};
		is.get( c );
		if ( !is ) {
			throw std::runtime_error( "nn_io: truncated stream (uint16)" );
		}
		v |= static_cast<std::uint16_t>( static_cast<unsigned char>( c ) )
		     << ( 8U * static_cast<unsigned>( i ) );
	}
	return v;
}

inline std::uint32_t read_u32_le( std::istream &is )
{
	std::uint32_t v = 0;
	for ( int i = 0; i < 4; ++i ) {
		char c{};
		is.get( c );
		if ( !is ) {
			throw std::runtime_error( "nn_io: truncated stream (uint32)" );
		}
		v |= static_cast<std::uint32_t>( static_cast<unsigned char>( c ) )
		     << ( 8U * static_cast<unsigned>( i ) );
	}
	return v;
}

inline std::uint64_t read_u64_le( std::istream &is )
{
	std::uint64_t v = 0;
	for ( int i = 0; i < 8; ++i ) {
		char c{};
		is.get( c );
		if ( !is ) {
			throw std::runtime_error( "nn_io: truncated stream (uint64)" );
		}
		v |= static_cast<std::uint64_t>( static_cast<unsigned char>( c ) )
		     << ( 8U * static_cast<unsigned>( i ) );
	}
	return v;
}

inline void write_kv( std::ostream &os, std::string const &key, std::vector<std::byte> const &value )
{
	if ( key.size() > 0xFFFFU ) {
		throw std::runtime_error( "nn_io: key too long" );
	}
	write_u16_le( os, static_cast<std::uint16_t>( key.size() ) );
	os.write( key.data(), static_cast<std::streamsize>( key.size() ) );
	if ( value.size() > 0xFFFFFFFFU ) {
		throw std::runtime_error( "nn_io: value too long" );
	}
	write_u32_le( os, static_cast<std::uint32_t>( value.size() ) );
	if ( !value.empty() ) {
		os.write( reinterpret_cast<char const *>( value.data() ),
		          static_cast<std::streamsize>( value.size() ) );
	}
}

inline void write_kv_ascii( std::ostream &os, std::string const &key, std::string const &ascii )
{
	std::vector<std::byte> v( reinterpret_cast<std::byte const *>( ascii.data() ),
	                          reinterpret_cast<std::byte const *>( ascii.data() + ascii.size() ) );
	write_kv( os, key, v );
}

inline void write_kv_empty( std::ostream &os, std::string const &key )
{
	write_kv( os, key, {} );
}

inline std::pair<std::string, std::vector<std::byte>> read_kv( std::istream &is )
{
	std::uint16_t const kl = read_u16_le( is );
	std::string key( static_cast<std::size_t>( kl ), '\0' );
	if ( kl > 0U ) {
		is.read( key.data(), static_cast<std::streamsize>( kl ) );
		if ( !is || static_cast<std::uint64_t>( is.gcount() ) != kl ) {
			throw std::runtime_error( "nn_io: truncated key bytes" );
		}
	}
	std::uint32_t const vl = read_u32_le( is );
	std::vector<std::byte> val( static_cast<std::size_t>( vl ) );
	if ( vl > 0U ) {
		is.read( reinterpret_cast<char *>( val.data() ), static_cast<std::streamsize>( vl ) );
		if ( !is || static_cast<std::uint64_t>( is.gcount() ) != vl ) {
			throw std::runtime_error( "nn_io: truncated value bytes" );
		}
	}
	return { std::move( key ), std::move( val ) };
}

inline std::string ascii_string( std::vector<std::byte> const &v )
{
	return { reinterpret_cast<char const *>( v.data() ), v.size() };
}

inline std::uint32_t parse_ascii_u32( std::vector<std::byte> const &v )
{
	return static_cast<std::uint32_t>( std::stoul( ascii_string( v ) ) );
}

inline std::int32_t parse_ascii_i32( std::vector<std::byte> const &v )
{
	return static_cast<std::int32_t>( std::stol( ascii_string( v ) ) );
}

inline std::uint64_t parse_ascii_u64( std::vector<std::byte> const &v )
{
	return static_cast<std::uint64_t>( std::stoull( ascii_string( v ) ) );
}

inline double read_double_bin( std::vector<std::byte> const &v )
{
	if ( v.size() != sizeof( double ) ) {
		throw std::runtime_error( "nn_io: bad double size" );
	}
	double d{};
	std::memcpy( &d, v.data(), sizeof d );
	return d;
}

inline void write_double_kv( std::ostream &os, std::string const &key, double d )
{
	std::vector<std::byte> val( sizeof( double ) );
	std::memcpy( val.data(), &d, sizeof d );
	write_kv( os, key, val );
}

inline void append_le_u64( std::vector<std::byte> &out, std::uint64_t v )
{
	for ( int i = 0; i < 8; ++i ) {
		out.push_back( static_cast<std::byte>( ( v >> ( 8U * i ) ) & 0xFFU ) );
	}
}

inline std::uint64_t read_le_u64_from_bytes( std::byte const *p )
{
	std::uint64_t v = 0;
	for ( int i = 0; i < 8; ++i ) {
		v |= static_cast<std::uint64_t>( static_cast<unsigned char>( p[i] ) )
		     << ( 8U * i );
	}
	return v;
}

inline std::tuple<DType, std::uint64_t, std::vector<std::byte>> parse_prefixed_blob_le(
    std::vector<std::byte> const &whole, DType header_dtype )
{
	if ( whole.size() < 1U + 8U ) {
		throw std::runtime_error( "nn_io: blob too small" );
	}
	DType const inner = static_cast<DType>( static_cast<unsigned char>( whole[0] ) );
	if ( inner != header_dtype ) {
		throw std::runtime_error( "nn_io: blob DType mismatch with file header" );
	}
	std::uint64_t const count = read_le_u64_from_bytes( whole.data() + 1 );
	std::size_t const es = elementSize( inner );
	if ( whole.size() < 1U + 8U + count * es ) {
		throw std::runtime_error( "nn_io: blob truncated" );
	}
	if ( whole.size() != 1U + 8U + count * es ) {
		throw std::runtime_error( "nn_io: blob tail garbage" );
	}
	std::vector<std::byte> payload( whole.begin() + 9, whole.end() );
	return { inner, count, std::move( payload ) };
}

template <typename T>
void write_tensor_blob_kv( std::ostream &os, std::string const &key, DType file_dt, T const *data,
                           std::size_t count )
{
	std::vector<std::byte> elements;
	convertNetworkBlobToHostFile<T>( data, count, file_dt, elements );
	std::vector<std::byte> value;
	value.push_back(
	    static_cast<std::byte>( static_cast<std::underlying_type_t<DType>>( file_dt ) ) );
	append_le_u64( value, static_cast<std::uint64_t>( count ) );
	value.insert( value.end(), elements.begin(), elements.end() );
	write_kv( os, key, value );
}

template <typename T>
void read_tensor_blob_into( std::vector<std::byte> const &whole, DType header_dtype, T *dst,
                            std::size_t expect_count )
{
	auto const [inner, cnt, payload] = parse_prefixed_blob_le( whole, header_dtype );
	if ( inner != header_dtype ) {
		(void)inner;
	}
	if ( cnt != expect_count ) {
		throw std::runtime_error( "nn_io: tensor element count mismatch" );
	}
	convertBlobHostToNetwork<T>( payload, header_dtype, dst, expect_count );
}

inline void write_file_header( std::ostream &os, Header const &h )
{
	os.write( h.magic, 4 );
	char V = static_cast<char>( h.version );
	os.put( V );
	os.put( static_cast<char>( static_cast<std::underlying_type_t<DType>>( h.dtype ) ) );
	write_u32_le( os, h.num_layers );
	write_u32_le( os, h.nn_type );
}

inline Header read_file_header( std::istream &is )
{
	Header h{};
	is.read( h.magic, 4 );
	if ( !is || is.gcount() != 4 ) {
		throw std::runtime_error( "nn_io: missing magic" );
	}
	char v{};
	is.get( v );
	if ( !is ) {
		throw std::runtime_error( "nn_io: missing version" );
	}
	h.version = static_cast<std::uint8_t>( v );
	char dt{};
	is.get( dt );
	if ( !is ) {
		throw std::runtime_error( "nn_io: missing dtype" );
	}
	h.dtype = static_cast<DType>( static_cast<unsigned char>( dt ) );
	h.num_layers = read_u32_le( is );
	h.nn_type    = read_u32_le( is );
	if ( !is ) {
		throw std::runtime_error( "nn_io: truncated header" );
	}
	if ( h.magic[0] != 'N' || h.magic[1] != 'N' || h.magic[2] != 'C' || h.magic[3] != 'X' ) {
		throw std::runtime_error( "nn_io: bad magic" );
	}
	if ( h.version != kNnFormatVersion ) {
		throw std::runtime_error( "nn_io: unsupported format version" );
	}
	return h;
}

inline void read_layer_map( std::istream &is, std::map<std::string, std::vector<std::byte>> &out )
{
	out.clear();
	auto kv0 = read_kv( is );
	if ( kv0.first != "LAYER_START" ) {
		throw std::runtime_error( "nn_io: expected LAYER_START" );
	}
	for ( ;; ) {
		auto kv = read_kv( is );
		if ( kv.first == "LAYER_END" ) {
			break;
		}
		out.insert_or_assign( std::move( kv.first ), std::move( kv.second ) );
	}
}

inline void require_key( std::map<std::string, std::vector<std::byte>> const &m,
                         std::string const &k )
{
	if ( m.find( k ) == m.end() ) {
		throw std::runtime_error( "nn_io: missing key: " + k );
	}
}

#if NEURAL_CUDA_ENABLED
template <typename T>
inline void cuda_memcpy_d2h( T const *dev, std::size_t n, std::vector<T> &out )
{
	out.resize( n );
	if ( n == 0U ) {
		return;
	}
	if ( cudaMemcpy( out.data(), dev, n * sizeof( T ), cudaMemcpyDeviceToHost ) != cudaSuccess ) {
		throw std::runtime_error( "nn_io: cudaMemcpy device->host failed" );
	}
}

template <typename T>
inline void cuda_memcpy_h2d( T const *host, std::size_t n, T *&dev_out )
{
	dev_out = nullptr;
	if ( n == 0U ) {
		return;
	}
	T *d = nullptr;
	if ( cudaMalloc( reinterpret_cast<void **>( &d ), n * sizeof( T ) ) != cudaSuccess ) {
		throw std::runtime_error( "nn_io: cudaMalloc failed" );
	}
	if ( cudaMemcpy( d, host, n * sizeof( T ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
		(void)cudaFree( d );
		throw std::runtime_error( "nn_io: cudaMemcpy host->device failed" );
	}
	dev_out = d;
}
#endif

} // namespace nn_io_detail

#include "nn_io_body.inl"

} // namespace refactor
} // namespace neural

#endif // REFACTOR_NN_IO_HPP
