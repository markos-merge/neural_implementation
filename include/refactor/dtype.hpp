#ifndef REFACTOR_DTYPE_HPP
#define REFACTOR_DTYPE_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#if NEURAL_CUDA_ENABLED
#include <cuda_fp16.h>
#if __has_include( <cuda_fp8.h> )
#include <cuda_fp8.h>
#define NEURAL_CUDA_FP8_HEADER 1
#endif
#endif

namespace neural {
namespace refactor {

enum class DType : std::uint8_t {
	Float32 = 0,
	Float64 = 1,
	Float16 = 2,
	Fp8E4M3 = 3,
	Fp8E5M2 = 4,
};

template <typename T>
struct dtype_traits;

template <>
struct dtype_traits<float> {
	static constexpr DType value = DType::Float32;
};

template <>
struct dtype_traits<double> {
	static constexpr DType value = DType::Float64;
};

#if NEURAL_CUDA_ENABLED
template <>
struct dtype_traits<__half> {
	static constexpr DType value = DType::Float16;
};
#if NEURAL_CUDA_FP8_HEADER
template <>
struct dtype_traits<__nv_fp8_e4m3> {
	static constexpr DType value = DType::Fp8E4M3;
};
template <>
struct dtype_traits<__nv_fp8_e5m2> {
	static constexpr DType value = DType::Fp8E5M2;
};
#endif
#endif

template <typename T>
inline constexpr DType dtype_of = dtype_traits<T>::value;

inline std::size_t elementSize( DType dt )
{
	switch ( dt ) {
	case DType::Float32:
		return sizeof( float );
	case DType::Float64:
		return sizeof( double );
#if NEURAL_CUDA_ENABLED
	case DType::Float16:
		return sizeof( __half );
#if NEURAL_CUDA_FP8_HEADER
	case DType::Fp8E4M3:
		return sizeof( __nv_fp8_e4m3 );
	case DType::Fp8E5M2:
		return sizeof( __nv_fp8_e5m2 );
#else
	case DType::Fp8E4M3:
	case DType::Fp8E5M2:
		throw std::logic_error( "elementSize: fp8 not available (no cuda_fp8.h)" );
#endif
#else
	case DType::Float16:
	case DType::Fp8E4M3:
	case DType::Fp8E5M2:
		throw std::logic_error( "elementSize: half/fp8 requires NEURAL_CUDA_ENABLED" );
#endif
	default:
		throw std::logic_error( "elementSize: unknown DType" );
	}
}

inline double readBlobElementAsDouble( void const *buf, DType file )
{
	switch ( file ) {
	case DType::Float32: {
		float x;
		std::memcpy( &x, buf, sizeof x );
		return static_cast<double>( x );
	}
	case DType::Float64: {
		double x;
		std::memcpy( &x, buf, sizeof x );
		return x;
	}
#if NEURAL_CUDA_ENABLED
	case DType::Float16: {
		__half h;
		std::memcpy( &h, buf, sizeof h );
		return static_cast<double>( __half2float( h ) );
	}
#if NEURAL_CUDA_FP8_HEADER
	case DType::Fp8E4M3: {
		__nv_fp8_e4m3 v;
		std::memcpy( &v, buf, sizeof v );
		float f{};
		f = static_cast<float>( v );
		return static_cast<double>( f );
	}
	case DType::Fp8E5M2: {
		__nv_fp8_e5m2 v;
		std::memcpy( &v, buf, sizeof v );
		float f{};
		f = static_cast<float>( v );
		return static_cast<double>( f );
	}
#endif
#endif
	default:
		break;
	}
	throw std::runtime_error( "readBlobElementAsDouble: unsupported file DType" );
}

template <typename T>
void writeElementToBufferAsDType( void *dst, T value, DType file_dt )
{
	if ( file_dt == dtype_of<T> ) {
		std::memcpy( dst, &value, sizeof( T ) );
		return;
	}
	double const d = static_cast<double>( value );
	switch ( file_dt ) {
	case DType::Float32: {
		float const x = static_cast<float>( d );
		std::memcpy( dst, &x, sizeof x );
		return;
	}
	case DType::Float64:
		std::memcpy( dst, &d, sizeof d );
		return;
#if NEURAL_CUDA_ENABLED
	case DType::Float16: {
		__half const h = __float2half( static_cast<float>( d ) );
		std::memcpy( dst, &h, sizeof h );
		return;
	}
#if NEURAL_CUDA_FP8_HEADER
	case DType::Fp8E4M3: {
		__nv_fp8_e4m3 const v( static_cast<float>( d ) );
		std::memcpy( dst, &v, sizeof v );
		return;
	}
	case DType::Fp8E5M2: {
		__nv_fp8_e5m2 const v( static_cast<float>( d ) );
		std::memcpy( dst, &v, sizeof v );
		return;
	}
#endif
#endif
	default:
		break;
	}
	throw std::runtime_error( "writeElementToBufferAsDType: unsupported target DType" );
}

template <typename T>
void convertBlobHostToNetwork( std::vector<std::byte> const &file_bytes,
                                 DType file_dt,
                                 T *dst,
                                 std::size_t count )
{
	std::size_t const es = elementSize( file_dt );
	if ( count == 0U )
		return;
	if ( file_bytes.size() < count * es )
		throw std::runtime_error( "convertBlobHostToNetwork: truncated blob" );
	if ( file_dt == dtype_of<T> ) {
		std::memcpy( dst, file_bytes.data(), count * sizeof( T ) );
		return;
	}
	for ( std::size_t i = 0; i < count; ++i ) {
		double const dv = readBlobElementAsDouble( file_bytes.data() + i * es, file_dt );
		dst[i] = static_cast<T>( dv );
	}
}

template <typename T>
void convertNetworkBlobToHostFile( void const *src,
                                     std::size_t count,
                                     DType file_dt,
                                     std::vector<std::byte> &out )
{
	if ( dtype_of<T> == file_dt ) {
		out.resize( count * sizeof( T ) );
		std::memcpy( out.data(), src, count * sizeof( T ) );
		return;
	}
	std::size_t const es = elementSize( file_dt );
	out.resize( count * es );
	for ( std::size_t i = 0; i < count; ++i ) {
		T const v = static_cast<T const *>( src )[i];
		writeElementToBufferAsDType( out.data() + i * es, v, file_dt );
	}
}

} // namespace refactor
} // namespace neural

#endif // REFACTOR_DTYPE_HPP
