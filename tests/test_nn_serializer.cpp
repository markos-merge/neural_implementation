#include "refactor/nn_io.hpp"
#include "refactor/sequential_nn.hpp"
#include "refactor/linear_layer.hpp"
#include "refactor/relu_layer.hpp"
#include "refactor/sigmoid_layer.hpp"
#include "refactor/softmax_layer.hpp"
#include "refactor/dropout_layer.hpp"
#include "refactor/max_pool_layer.hpp"
#include "refactor/batch_norm_layer.hpp"
#include "refactor/convolutional_layer.hpp"
#include "refactor/network.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#if NEURAL_CUDA_ENABLED
#include "neural_cuda_runtime.hpp"
#include <cuda_runtime.h>
#endif

using neural::refactor::BatchNormLayer;
using neural::refactor::ConvolutionalLayer;
using neural::refactor::Cpu;
using neural::refactor::DropoutLayer;
using neural::refactor::LinearLayer;
using neural::refactor::load;
using neural::refactor::MaxPoolLayer;
using neural::refactor::ReLULayer;
using neural::refactor::save;
using neural::refactor::SequentialNN2;
using neural::refactor::SequentialNetwork;
using neural::refactor::SigmoidLayer;
#if NEURAL_CUDA_ENABLED
using neural::refactor::Cuda;
#endif

using Catch::Matchers::WithinAbs;

namespace {

std::string unique_tmp( char const *base )
{
	auto const dir = std::filesystem::temp_directory_path();
	return ( dir / ( std::string( "nnio_" ) + base + "_" + std::to_string( std::rand() ) ) )
	    .string();
}

#if NEURAL_CUDA_ENABLED
void cuda_d2h_float( float const *dev, std::size_t n, std::vector<float> &out )
{
	out.resize( n );
	if ( n == 0U ) {
		return;
	}
	if ( cudaMemcpy( out.data(), dev, n * sizeof( float ), cudaMemcpyDeviceToHost ) != cudaSuccess ) {
		throw std::runtime_error( "cudaMemcpy device->host failed" );
	}
}
#endif

} // namespace

TEST_CASE( "nn_io linear round-trip", "[refactor][nn_io]" )
{
	std::string const path = unique_tmp( "lin" );
	SequentialNN2<float> net;
	net.addLayer( LinearLayer<float>{ 3U } );
	net.wire();

	std::vector<float> in( 2 * 5, 1.0F );
	net.forward( in.data(), 2, 5 );

	net.forEachLayer( []( auto &l ) {
		l.layerName( "fc" );
		l.layerId( 7 );
	} );

	save( net, path );

	SequentialNN2<float> net2;
	load( net2, path );
	net2.wire();

	REQUIRE( net2.numLayers() == 1 );

	LinearLayer<float> *p0 = nullptr;
	LinearLayer<float> *p1 = nullptr;
	net.forEachLayer( [&]( auto &l ) {
		if constexpr ( std::is_same_v<std::decay_t<decltype( l )>, LinearLayer<float>> ) {
			p0 = &l;
		}
	} );
	net2.forEachLayer( [&]( auto &l ) {
		if constexpr ( std::is_same_v<std::decay_t<decltype( l )>, LinearLayer<float>> ) {
			p1 = &l;
		}
	} );
	REQUIRE( p0 != nullptr );
	REQUIRE( p1 != nullptr );
	REQUIRE( p0->layerName() == p1->layerName() );
	REQUIRE( p0->layerId() == p1->layerId() );
	REQUIRE( p0->numWeightParams() == p1->numWeightParams() );

	for ( std::size_t i = 0; i < p0->numWeightParams(); ++i ) {
		REQUIRE( std::abs( p0->getWeights()[i] - p1->getWeights()[i] ) < 1e-6F );
	}
	for ( std::size_t i = 0; i < p0->numBiasParams(); ++i ) {
		REQUIRE( std::abs( p0->getBias()[i] - p1->getBias()[i] ) < 1e-6F );
	}

	std::filesystem::remove( path );
}

TEST_CASE( "nn_io loaded network forward backward matches reference", "[refactor][nn_io]" )
{
	std::string const path = unique_tmp( "runfwd" );
	SequentialNN2<float> net;
	net.addLayer( LinearLayer<float>{ 6 } );
	net.addLayer( ReLULayer<float>{} );
	net.wire();

	constexpr std::size_t  N           = 4;
	constexpr std::size_t  in_features = 11;
	std::vector<float>     x( N * in_features );
	for ( std::size_t i = 0; i < x.size(); ++i )
		x[i] = -0.3F + 0.07F * static_cast<float>( static_cast<int>( i % 17 ) );

	net.forward( x.data(), N, in_features );
	save( net, path );

	SequentialNN2<float> net2;
	load( net2, path );
	// load() already calls wire(); do not wire() again (second wiring breaks output latch shape).

	REQUIRE( net2.numLayers() == 2 );

	REQUIRE_NOTHROW( net.forward( x.data(), N, in_features ) );
	REQUIRE_NOTHROW( net2.forward( x.data(), N, in_features ) );
	REQUIRE( net.outputSize() == net2.outputSize() );
	REQUIRE( net2.outputSize() == N * 6 );

	float constexpr          eps = 1e-5F;
	float const *const       y0 = net.output();
	float const *const       y1 = net2.output();
	for ( std::size_t i = 0; i < net.outputSize(); ++i ) {
		REQUIRE( std::isfinite( y1[i] ) );
		REQUIRE_THAT( y0[i], WithinAbs( y1[i], eps ) );
	}

	std::vector<float> upstream( N * 6 );
	for ( std::size_t i = 0; i < upstream.size(); ++i )
		upstream[i] = -0.11F + 0.03F * static_cast<float>( static_cast<int>( i % 7 ) );

	REQUIRE_NOTHROW( net.backward( upstream.data() ) );
	REQUIRE_NOTHROW( net2.backward( upstream.data() ) );

	LinearLayer<float> *lin0 = nullptr;
	LinearLayer<float> *lin1 = nullptr;
	net.forEachLayer( [&]( auto &l ) {
		if constexpr ( std::is_same_v<std::decay_t<decltype( l )>, LinearLayer<float>> )
			lin0 = &l;
	} );
	net2.forEachLayer( [&]( auto &l ) {
		if constexpr ( std::is_same_v<std::decay_t<decltype( l )>, LinearLayer<float>> )
			lin1 = &l;
	} );
	REQUIRE( lin0 != nullptr );
	REQUIRE( lin1 != nullptr );

	float constexpr geps = 2e-4F;
	for ( std::size_t i = 0; i < lin0->numWeightParams(); ++i ) {
		REQUIRE( std::isfinite( lin1->getGradWeights()[i] ) );
		REQUIRE_THAT( lin0->getGradWeights()[i], WithinAbs( lin1->getGradWeights()[i], geps ) );
	}
	for ( std::size_t i = 0; i < lin0->numBiasParams(); ++i ) {
		REQUIRE( std::isfinite( lin1->getGradBias()[i] ) );
		REQUIRE_THAT( lin0->getGradBias()[i], WithinAbs( lin1->getGradBias()[i], geps ) );
	}

	std::filesystem::remove( path );
}

TEST_CASE( "nn_io relu meta round-trip", "[refactor][nn_io]" )
{
	std::string const path = unique_tmp( "relu" );
	SequentialNN2<float> net;
	net.addLayer( ReLULayer<float>{} );
	net.wire();
	net.forEachLayer( []( auto &l ) {
		l.layerName( "relu1" );
		l.layerId( 1 );
	} );
	save( net, path );

	SequentialNN2<float> net2;
	load( net2, path );
	net2.wire();
	net2.forEachLayer( []( auto &l ) {
		REQUIRE( l.layerName() == "relu1" );
		REQUIRE( l.layerId() == 1 );
	} );
	std::filesystem::remove( path );
}

TEST_CASE( "SequentialNetwork save overload", "[refactor][nn_io]" )
{
	std::string const path = unique_tmp( "sn" );
	SequentialNetwork<float> sn;
	sn.nn().addLayer( SigmoidLayer<float>{} );
	sn.nn().wire();
	sn.nn().forEachLayer( []( auto &l ) {
		l.layerName( "sig" );
		l.layerId( 2 );
	} );
	save( sn, path );

	SequentialNetwork<float> sn2;
	load( sn2, path );
	sn2.nn().wire();
	sn2.nn().forEachLayer( []( auto &l ) {
		REQUIRE( l.layerName() == "sig" );
	} );
	std::filesystem::remove( path );
}

TEST_CASE( "nn_io conv sized ctor round-trip", "[refactor][nn_io]" )
{
	std::string const path = unique_tmp( "conv" );
	std::size_t const C   = 2;
	std::size_t const H   = 5;
	std::size_t const W   = 5;
	SequentialNN2<float> net;
	net.addLayer( ConvolutionalLayer<float, Cpu>{
	    3, 3, std::array<std::size_t, 3>{ C, H, W }, 1, 1 } );
	net.wire();

	std::vector<float> in( 1 * C * H * W, 0.1F );
	net.forward( in.data(), 1, C * H * W );

	std::array<std::size_t, 3> const expect_chw{ C, H, W };
	ConvolutionalLayer<float, Cpu> *conv0 = nullptr;
	net.forEachLayer( [&]( auto &l ) {
		if constexpr ( std::is_same_v<std::decay_t<decltype( l )>, ConvolutionalLayer<float, Cpu>> )
			conv0 = &l;
	} );
	REQUIRE( conv0 != nullptr );
	REQUIRE( conv0->inputChw().has_value() );
	REQUIRE( *conv0->inputChw() == expect_chw );

	net.forEachLayer( []( auto &l ) {
		l.layerName( "cv" );
		l.layerId( 9 );
	} );

	save( net, path );

	SequentialNN2<float> net2;
	load( net2, path );
	REQUIRE( net2.numLayers() == 1 );

	ConvolutionalLayer<float, Cpu> *conv1 = nullptr;
	net2.forEachLayer( [&]( auto &l ) {
		if constexpr ( std::is_same_v<std::decay_t<decltype( l )>, ConvolutionalLayer<float, Cpu>> )
			conv1 = &l;
	} );
	REQUIRE( conv1 != nullptr );
	REQUIRE( conv1->inputChw().has_value() );
	REQUIRE( *conv1->inputChw() == expect_chw );

	net.forward( in.data(), 1, C * H * W );
	net2.forward( in.data(), 1, C * H * W );
	REQUIRE( net.outputSize() == net2.outputSize() );

	float constexpr          feps = 1e-5F;
	std::vector<float>       y0( net.outputSize() ), y1( net2.outputSize() );
	std::memcpy( y0.data(), net.output(), net.outputSize() * sizeof( float ) );
	std::memcpy( y1.data(), net2.output(), net2.outputSize() * sizeof( float ) );
	for ( std::size_t i = 0; i < y0.size(); ++i ) {
		REQUIRE_THAT( y0[i], WithinAbs( y1[i], feps ) );
	}

	std::filesystem::remove( path );
}

TEST_CASE( "nn_io dropout round-trip", "[refactor][nn_io]" )
{
	std::string const path = unique_tmp( "do" );
	SequentialNN2<float> net;
	net.addLayer( DropoutLayer<float>{ 0.5F, 12345U } );
	net.wire();
	net.forEachLayer( []( auto &l ) {
		l.layerName( "drop" );
		l.layerId( 3 );
	} );
	save( net, path );

	SequentialNN2<float> net2;
	load( net2, path );
	net2.wire();
	net2.forEachLayer( []( auto &l ) {
		REQUIRE( l.layerName() == "drop" );
	} );
	std::filesystem::remove( path );
}

TEST_CASE( "nn_io maxpool round-trip", "[refactor][nn_io]" )
{
	std::string const path = unique_tmp( "mp" );
	SequentialNN2<float> net;
	net.addLayer( MaxPoolLayer<float, Cpu>{ 2, 2 } );
	net.wire();
	net.forEachLayer( []( auto &l ) {
		l.layerName( "mp" );
		l.layerId( 4 );
	} );
	save( net, path );

	SequentialNN2<float> net2;
	load( net2, path );
	net2.wire();
	REQUIRE( net2.numLayers() == 1 );
	std::filesystem::remove( path );
}

TEST_CASE( "nn_io batchnorm pointer ctor round-trip", "[refactor][nn_io]" )
{
	std::filesystem::path const p = unique_tmp( "bn" );
	std::size_t const            C = 3;
	std::vector<float>           g( C, 1.0F );
	std::vector<float>           b( C, 0.0F );
	std::vector<float>           rm( C, 0.0F );
	std::vector<float>           rv( C, 1.0F );
	float *                      pg = new float[C];
	float *                      pb = new float[C];
	float *                      pm = new float[C];
	float *                      pv = new float[C];
	std::memcpy( pg, g.data(), C * sizeof( float ) );
	std::memcpy( pb, b.data(), C * sizeof( float ) );
	std::memcpy( pm, rm.data(), C * sizeof( float ) );
	std::memcpy( pv, rv.data(), C * sizeof( float ) );

	SequentialNN2<float> net;
	net.addLayer(
	    BatchNormLayer<float, Cpu>{ pg, pb, pm, pv, C, 1e-5F, 0.1F } );
	net.wire();
	net.forEachLayer( []( auto &l ) {
		l.layerName( "bn" );
		l.layerId( 5 );
	} );

	save( net, p.string() );

	SequentialNN2<float> net2;
	load( net2, p.string() );
	net2.wire();
	REQUIRE( net2.numLayers() == 1 );
	std::filesystem::remove( p );
}

#if NEURAL_CUDA_ENABLED

TEST_CASE( "nn_io linear round-trip (cuda)", "[refactor][nn_io][cuda]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::string const           path = unique_tmp( "lin_cu" );
	SequentialNN2<float, Cuda> net;
	net.addLayer( LinearLayer<float, Cuda>{ 3U } );
	net.wire();

	std::vector<float> in( 2 * 5, 1.0F );
	net.forward( in.data(), 2, 5 );

	net.forEachLayer( []( auto &l ) {
		l.layerName( "fc" );
		l.layerId( 7 );
	} );

	save( net, path );

	SequentialNN2<float, Cuda> net2;
	load( net2, path );

	REQUIRE( net2.numLayers() == 1 );

	LinearLayer<float, Cuda> *p0 = nullptr;
	LinearLayer<float, Cuda> *p1 = nullptr;
	net.forEachLayer( [&]( auto &l ) {
		if constexpr ( std::is_same_v<std::decay_t<decltype( l )>, LinearLayer<float, Cuda>> ) {
			p0 = &l;
		}
	} );
	net2.forEachLayer( [&]( auto &l ) {
		if constexpr ( std::is_same_v<std::decay_t<decltype( l )>, LinearLayer<float, Cuda>> ) {
			p1 = &l;
		}
	} );
	REQUIRE( p0 != nullptr );
	REQUIRE( p1 != nullptr );
	REQUIRE( p0->layerName() == p1->layerName() );
	REQUIRE( p0->layerId() == p1->layerId() );
	REQUIRE( p0->numWeightParams() == p1->numWeightParams() );

	std::vector<float> w0, w1, b0, b1;
	cuda_d2h_float( p0->getWeights(), p0->numWeightParams(), w0 );
	cuda_d2h_float( p1->getWeights(), p1->numWeightParams(), w1 );
	cuda_d2h_float( p0->getBias(), p0->numBiasParams(), b0 );
	cuda_d2h_float( p1->getBias(), p1->numBiasParams(), b1 );

	for ( std::size_t i = 0; i < w0.size(); ++i ) {
		REQUIRE( std::abs( w0[i] - w1[i] ) < 1e-6F );
	}
	for ( std::size_t i = 0; i < b0.size(); ++i ) {
		REQUIRE( std::abs( b0[i] - b1[i] ) < 1e-6F );
	}

	std::filesystem::remove( path );
}

TEST_CASE( "nn_io loaded network forward backward matches reference (cuda)",
           "[refactor][nn_io][cuda]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::string const path = unique_tmp( "runfwd_cu" );

	SequentialNN2<float, Cuda> net;
	net.addLayer( LinearLayer<float, Cuda>{ 6 } );
	net.addLayer( ReLULayer<float, Cuda>{} );
	net.wire();

	constexpr std::size_t  N           = 4;
	constexpr std::size_t  in_features = 11;
	std::vector<float>     x( N * in_features );
	for ( std::size_t i = 0; i < x.size(); ++i )
		x[i] = -0.3F + 0.07F * static_cast<float>( static_cast<int>( i % 17 ) );

	net.forward( x.data(), N, in_features );
	save( net, path );

	SequentialNN2<float, Cuda> net2;
	load( net2, path );

	REQUIRE( net2.numLayers() == 2 );

	REQUIRE_NOTHROW( net.forward( x.data(), N, in_features ) );
	REQUIRE_NOTHROW( net2.forward( x.data(), N, in_features ) );
	REQUIRE( net2.outputSize() == N * 6 );

	std::vector<float> y0( net.outputSize() ), y1( net2.outputSize() );
	cuda_d2h_float( net.output(), net.outputSize(), y0 );
	cuda_d2h_float( net2.output(), net2.outputSize(), y1 );

	float constexpr    eps = 1e-5F;
	for ( std::size_t i = 0; i < y0.size(); ++i ) {
		REQUIRE( std::isfinite( y1[i] ) );
		REQUIRE_THAT( y0[i], WithinAbs( y1[i], eps ) );
	}

	std::vector<float> upstream( N * 6 );
	for ( std::size_t i = 0; i < upstream.size(); ++i )
		upstream[i] = -0.11F + 0.03F * static_cast<float>( static_cast<int>( i % 7 ) );

	REQUIRE_NOTHROW( net.backward( upstream.data() ) );
	REQUIRE_NOTHROW( net2.backward( upstream.data() ) );

	LinearLayer<float, Cuda> *lin0 = nullptr;
	LinearLayer<float, Cuda> *lin1 = nullptr;
	net.forEachLayer( [&]( auto &l ) {
		if constexpr ( std::is_same_v<std::decay_t<decltype( l )>, LinearLayer<float, Cuda>> )
			lin0 = &l;
	} );
	net2.forEachLayer( [&]( auto &l ) {
		if constexpr ( std::is_same_v<std::decay_t<decltype( l )>, LinearLayer<float, Cuda>> )
			lin1 = &l;
	} );
	REQUIRE( lin0 != nullptr );
	REQUIRE( lin1 != nullptr );

	std::vector<float> gw0, gw1, gb0, gb1;
	cuda_d2h_float( lin0->getGradWeights(), lin0->numWeightParams(), gw0 );
	cuda_d2h_float( lin1->getGradWeights(), lin1->numWeightParams(), gw1 );
	cuda_d2h_float( lin0->getGradBias(), lin0->numBiasParams(), gb0 );
	cuda_d2h_float( lin1->getGradBias(), lin1->numBiasParams(), gb1 );

	float constexpr geps = 2e-4F;
	for ( std::size_t i = 0; i < gw0.size(); ++i ) {
		REQUIRE( std::isfinite( gw1[i] ) );
		REQUIRE_THAT( gw0[i], WithinAbs( gw1[i], geps ) );
	}
	for ( std::size_t i = 0; i < gb0.size(); ++i ) {
		REQUIRE( std::isfinite( gb1[i] ) );
		REQUIRE_THAT( gb0[i], WithinAbs( gb1[i], geps ) );
	}

	std::filesystem::remove( path );
}

TEST_CASE( "nn_io relu meta round-trip (cuda)", "[refactor][nn_io][cuda]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::string const path = unique_tmp( "relu_cu" );
	SequentialNN2<float, Cuda> net;
	net.addLayer( ReLULayer<float, Cuda>{} );
	net.wire();
	net.forEachLayer( []( auto &l ) {
		l.layerName( "relu1" );
		l.layerId( 1 );
	} );
	save( net, path );

	SequentialNN2<float, Cuda> net2;
	load( net2, path );
	net2.forEachLayer( []( auto &l ) {
		REQUIRE( l.layerName() == "relu1" );
		REQUIRE( l.layerId() == 1 );
	} );
	std::filesystem::remove( path );
}

TEST_CASE( "SequentialNetwork save overload (cuda)", "[refactor][nn_io][cuda]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::string const path = unique_tmp( "sn_cu" );
	SequentialNetwork<float, Cuda> sn;
	sn.nn().addLayer( SigmoidLayer<float, Cuda>{} );
	sn.nn().wire();
	sn.nn().forEachLayer( []( auto &l ) {
		l.layerName( "sig" );
		l.layerId( 2 );
	} );
	save( sn, path );

	SequentialNetwork<float, Cuda> sn2;
	load( sn2, path );
	sn2.nn().forEachLayer( []( auto &l ) {
		REQUIRE( l.layerName() == "sig" );
	} );
	std::filesystem::remove( path );
}

#if NEURAL_CUDNN_ENABLED

TEST_CASE( "nn_io conv sized ctor round-trip (cuda)", "[refactor][nn_io][cuda]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::string const path = unique_tmp( "conv_cu" );
	std::size_t const C  = 2;
	std::size_t const H  = 5;
	std::size_t const W  = 5;
	SequentialNN2<float, Cuda> net;
	net.addLayer( ConvolutionalLayer<float, Cuda>{
	    3, 3, std::array<std::size_t, 3>{ C, H, W }, 1, 1 } );
	net.wire();

	std::vector<float> in( 1 * C * H * W, 0.1F );
	net.forward( in.data(), 1, C * H * W );

	std::array<std::size_t, 3> const expect_chw{ C, H, W };
	ConvolutionalLayer<float, Cuda> *conv0 = nullptr;
	net.forEachLayer( [&]( auto &l ) {
		if constexpr ( std::is_same_v<std::decay_t<decltype( l )>, ConvolutionalLayer<float, Cuda>> )
			conv0 = &l;
	} );
	REQUIRE( conv0 != nullptr );
	REQUIRE( conv0->inputChw().has_value() );
	REQUIRE( *conv0->inputChw() == expect_chw );

	net.forEachLayer( []( auto &l ) {
		l.layerName( "cv" );
		l.layerId( 9 );
	} );

	save( net, path );

	SequentialNN2<float, Cuda> net2;
	load( net2, path );
	REQUIRE( net2.numLayers() == 1 );

	ConvolutionalLayer<float, Cuda> *conv1 = nullptr;
	net2.forEachLayer( [&]( auto &l ) {
		if constexpr ( std::is_same_v<std::decay_t<decltype( l )>, ConvolutionalLayer<float, Cuda>> )
			conv1 = &l;
	} );
	REQUIRE( conv1 != nullptr );
	REQUIRE( conv1->inputChw().has_value() );
	REQUIRE( *conv1->inputChw() == expect_chw );

	net.forward( in.data(), 1, C * H * W );
	net2.forward( in.data(), 1, C * H * W );
	REQUIRE( net.outputSize() == net2.outputSize() );

	float constexpr    feps = 5e-4F;
	std::vector<float> y0, y1;
	cuda_d2h_float( net.output(), net.outputSize(), y0 );
	cuda_d2h_float( net2.output(), net2.outputSize(), y1 );
	for ( std::size_t i = 0; i < y0.size(); ++i ) {
		REQUIRE_THAT( y0[i], WithinAbs( y1[i], feps ) );
	}

	std::filesystem::remove( path );
}

#endif // NEURAL_CUDNN_ENABLED

TEST_CASE( "nn_io dropout round-trip (cuda)", "[refactor][nn_io][cuda]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::string const path = unique_tmp( "do_cu" );
	SequentialNN2<float, Cuda> net;
	net.addLayer( DropoutLayer<float, Cuda>{ 0.5F, 12345U } );
	net.wire();
	net.forEachLayer( []( auto &l ) {
		l.layerName( "drop" );
		l.layerId( 3 );
	} );
	save( net, path );

	SequentialNN2<float, Cuda> net2;
	load( net2, path );
	net2.forEachLayer( []( auto &l ) {
		REQUIRE( l.layerName() == "drop" );
	} );
	std::filesystem::remove( path );
}

TEST_CASE( "nn_io maxpool round-trip (cuda)", "[refactor][nn_io][cuda]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::string const path = unique_tmp( "mp_cu" );
	SequentialNN2<float, Cuda> net;
	net.addLayer( MaxPoolLayer<float, Cuda>{ 2, 2 } );
	net.wire();
	net.forEachLayer( []( auto &l ) {
		l.layerName( "mp" );
		l.layerId( 4 );
	} );
	save( net, path );

	SequentialNN2<float, Cuda> net2;
	load( net2, path );
	REQUIRE( net2.numLayers() == 1 );
	std::filesystem::remove( path );
}

TEST_CASE( "nn_io batchnorm default ctor round-trip (cuda)", "[refactor][nn_io][cuda]" )
{
	if ( !neural::cuda_runtime_ready() ) {
		SKIP( "No CUDA device available" );
	}
	std::filesystem::path const p      = unique_tmp( "bn_cu" );
	constexpr std::size_t         N    = 2;
	constexpr std::size_t         Cchan = 3;

	SequentialNN2<float, Cuda> net;
	net.addLayer( BatchNormLayer<float, Cuda>{ 1e-5F, 0.1F } );
	net.wire();

	std::vector<float> in( N * Cchan, 0.15F );
	net.forward( in.data(), N, Cchan );

	net.forEachLayer( []( auto &l ) {
		l.layerName( "bn" );
		l.layerId( 5 );
	} );

	save( net, p.string() );

	SequentialNN2<float, Cuda> net2;
	load( net2, p.string() );
	REQUIRE( net2.numLayers() == 1 );
	std::filesystem::remove( p );
}

#endif // NEURAL_CUDA_ENABLED
