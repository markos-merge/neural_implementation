// (aligned with PyTorch - (target * log_softmax).sum() / B and grad (p - t) / B).

#include "cross_entropy_softmax_loss.hpp"
#include "neural_cuda_runtime.hpp"
#include "tensor.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if NEURAL_CUDA_ENABLED
#include "cuda_tensor.hpp"
#endif

using neural::SoftmaxCrossEntropyLoss;
using neural::Tensor;
using Catch::Matchers::WithinAbs;

namespace {

constexpr float k_tol = 5e-5f;

std::filesystem::path data_dir()
{
	return std::filesystem::path( __FILE__ ).parent_path() / "data" / "pytorch_ref" / "softmax_cross_entropy";
}

std::vector<float> read_float_file( std::filesystem::path const &path )
{
	std::ifstream f( path, std::ios::binary );
	REQUIRE( f.good() );
	f.seekg( 0, std::ios::end );
	auto const n = static_cast<std::size_t>( f.tellg() );
	REQUIRE( n % sizeof( float ) == 0 );
	f.seekg( 0, std::ios::beg );
	std::vector<float> out( n / sizeof( float ) );
	f.read( reinterpret_cast<char *>( out.data() ),
	        static_cast<std::streamsize>( n ) );
	REQUIRE( static_cast<std::size_t>( f.gcount() ) == n );
	return out;
}

struct Meta
{
	std::size_t B = 0, C = 0;
};

Meta read_meta( std::filesystem::path const &p )
{
	std::ifstream f( p );
	REQUIRE( f.good() );
	std::string line;
	REQUIRE( std::getline( f, line ) );
	Meta m;
	std::size_t const pB = line.find( "B=" );
	std::size_t const pC = line.find( "C=" );
	REQUIRE( pB != std::string::npos );
	REQUIRE( pC != std::string::npos );
	std::size_t const b_start = pB + 2;
	std::size_t const spc     = line.find( ' ', b_start );
	m.B         = static_cast<std::size_t>( std::stoull( line.substr( b_start, spc - b_start ) ) );
	m.C         = static_cast<std::size_t>( std::stoull( line.substr( pC + 2 ) ) );
	return m;
}

} // namespace

TEST_CASE( "SoftmaxCrossEntropyLoss forward/backward matches PyTorch ref", "[pytorch_reference][softmax_ce][cpu]" )
{
	std::filesystem::path const d = data_dir();
	Meta const meta = read_meta( d / "meta.txt" );
	std::vector<float> const Lh  = read_float_file( d / "logits.bin" );
	std::vector<float> const t_h = read_float_file( d / "target.bin" );
	std::vector<float> const loss_r = read_float_file( d / "loss.bin" );
	std::vector<float> const g_ref  = read_float_file( d / "grad_logits.bin" );
	REQUIRE( loss_r.size() == 1u );
	REQUIRE( Lh.size() == meta.B * meta.C );
	REQUIRE( t_h.size() == Lh.size() );
	REQUIRE( g_ref.size() == Lh.size() );

	Tensor<float> logits( meta.B, meta.C, Lh.begin(), Lh.end() );
	Tensor<float> target( meta.B, meta.C, t_h.begin(), t_h.end() );

	SoftmaxCrossEntropyLoss<Tensor<float>> loss;
	Tensor<float> L = loss.forward( logits, target );
	REQUIRE( L.rows() == 1u );
	REQUIRE( L.cols() == 1u );
	REQUIRE_THAT( L( 0, 0 ), WithinAbs( loss_r[0], k_tol ) );

	Tensor<float> g = loss.backward();
	REQUIRE( g.rows() == meta.B );
	REQUIRE( g.cols() == meta.C );
	for ( std::size_t i = 0; i < g_ref.size(); ++i ) {
		REQUIRE_THAT( g.data()[i], WithinAbs( g_ref[i], k_tol ) );
	}
}

#if NEURAL_CUDA_ENABLED
TEST_CASE( "SoftmaxCrossEntropyLoss forward/backward matches PyTorch ref (CUDA)", "[pytorch_reference][softmax_ce][cuda]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::filesystem::path const d = data_dir();
	Meta const meta = read_meta( d / "meta.txt" );
	std::vector<float> const Lh  = read_float_file( d / "logits.bin" );
	std::vector<float> const t_h = read_float_file( d / "target.bin" );
	std::vector<float> const loss_r = read_float_file( d / "loss.bin" );
	std::vector<float> const g_ref  = read_float_file( d / "grad_logits.bin" );
	REQUIRE( loss_r.size() == 1u );
	REQUIRE( Lh.size() == meta.B * meta.C );
	REQUIRE( t_h.size() == Lh.size() );
	REQUIRE( g_ref.size() == Lh.size() );

	using T = float;
	neural::CudaTensor<T> logits( meta.B, meta.C, Lh.begin(), Lh.end() );
	neural::CudaTensor<T> target( meta.B, meta.C, t_h.begin(), t_h.end() );

	SoftmaxCrossEntropyLoss<neural::CudaTensor<T>> loss;
	neural::CudaTensor<T> L = loss.forward( logits, target );
	REQUIRE( L.rows() == 1u );
	REQUIRE( L.cols() == 1u );
	REQUIRE_THAT( L( 0, 0 ), WithinAbs( loss_r[0], k_tol ) );

	neural::CudaTensor<T> g = loss.backward();
	REQUIRE( g.rows() == meta.B );
	REQUIRE( g.cols() == meta.C );
	std::vector<T> g_host( g_ref.size() );
	g.copyToHost( g_host.data() );
	for ( std::size_t i = 0; i < g_ref.size(); ++i ) {
		REQUIRE_THAT( g_host[i], WithinAbs( g_ref[i], k_tol ) );
	}
}
#endif
