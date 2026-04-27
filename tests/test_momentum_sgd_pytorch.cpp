// Compares MomentumSGDOptimizer to PyTorch SGD (reference blobs in tests/data/pytorch_ref/).
// Reference is built by scripts/generate_pytorch_reference.py (momentum / momentum_nesterov).

#include "linear_layer.hpp"
#include "mse_loss.hpp"
#include "momentum_sgd_optimizer.hpp"
#include "sequential_nn.hpp"
#include "tensor.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <vector>

using neural::LinearLayer;
using neural::MSELoss;
using neural::MomentumSGDOptimizer;
using neural::SequentialNN;
using neural::Tensor;
using Catch::Matchers::WithinAbs;

namespace {

constexpr float k_tol = 5e-4f;

std::filesystem::path data_dir( char const *sub )
{
	return std::filesystem::path( __FILE__ ).parent_path() / "data" / "pytorch_ref" / sub;
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

void require_equal_tensor( Tensor<float> const &a, std::vector<float> const &ref, float eps )
{
	REQUIRE( a.size() == ref.size() );
	for ( std::size_t i = 0; i < ref.size(); ++i ) {
		REQUIRE_THAT( a.data()[i], WithinAbs( ref[i], eps ) );
	}
}

} // namespace

// Two full-batch SGD + momentum steps; data layout matches the generator.
TEST_CASE( "MomentumSGD matches PyTorch (plain) after two full-batch steps", "[pytorch_reference][momentum_sgd][cpu]" )
{
	char const *sub    = "momentum_sgd";
	std::vector<float> w_init  = read_float_file( data_dir( sub ) / "w_init.bin" );
	std::vector<float> b_init  = read_float_file( data_dir( sub ) / "b_init.bin" );
	std::vector<float> w_after = read_float_file( data_dir( sub ) / "w_after.bin" );
	std::vector<float> b_after = read_float_file( data_dir( sub ) / "b_after.bin" );
	std::vector<float> const x_h = read_float_file( data_dir( sub ) / "x.bin" );
	std::vector<float> const t_h = read_float_file( data_dir( sub ) / "t.bin" );
	REQUIRE( w_init.size() == 2u * 1u );
	REQUIRE( b_init.size() == 1u );
	REQUIRE( w_after.size() == 2u );
	REQUIRE( b_after.size() == 1u );
	REQUIRE( x_h.size() == 4u * 2u );
	REQUIRE( t_h.size() == 4u * 1u );

	Tensor<float> W0( 2, 1, w_init.begin(), w_init.end() );
	Tensor<float> B0( 1, 1, b_init.begin(), b_init.end() );
	using SimpleNN = SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>>;
	SimpleNN nn( LinearLayer<Tensor<float>>( W0, B0 ) );
	MomentumSGDOptimizer<Tensor<float>, SimpleNN> opt( nn );
	opt.m_learning_rate  = 0.02f;
	opt.m_momentum      = 0.9f;
	opt.m_l2_regularizer = 0.0f;
	opt.m_nesterov      = false;
	opt.m_epochs         = 2;
	opt.m_batch_size     = 4;

	std::vector<Tensor<float>> inputs;
	std::vector<Tensor<float>> targets;
	for ( int r = 0; r < 4; ++r ) {
		std::vector<float> row( 2u );
		row[0] = x_h[static_cast<std::size_t>( r * 2 )];
		row[1] = x_h[static_cast<std::size_t>( r * 2 + 1 )];
		inputs.push_back( Tensor<float>( 1, 2, row.begin(), row.end() ) );
		targets.push_back( Tensor<float>(
		    1, 1, &t_h[static_cast<std::size_t>( r )], &t_h[static_cast<std::size_t>( r )] + 1 ) );
	}
	opt.train( inputs, targets );

	LinearLayer<Tensor<float>> *L = nullptr;
	nn.forEachLayer( [&]( auto &layer ) { L = &layer; } );
	REQUIRE( L != nullptr );
	require_equal_tensor( L->getWeights(), w_after, k_tol );
	require_equal_tensor( L->getBias(), b_after, k_tol );
}

TEST_CASE( "MomentumSGD matches PyTorch (Nesterov) after two full-batch steps", "[pytorch_reference][momentum_sgd][cpu]" )
{
	char const *sub    = "momentum_sgd_nesterov";
	std::vector<float> w_init  = read_float_file( data_dir( sub ) / "w_init.bin" );
	std::vector<float> b_init  = read_float_file( data_dir( sub ) / "b_init.bin" );
	std::vector<float> w_after = read_float_file( data_dir( sub ) / "w_after.bin" );
	std::vector<float> b_after = read_float_file( data_dir( sub ) / "b_after.bin" );
	std::vector<float> const x_h = read_float_file( data_dir( sub ) / "x.bin" );
	std::vector<float> const t_h = read_float_file( data_dir( sub ) / "t.bin" );
	Tensor<float> W0( 2, 1, w_init.begin(), w_init.end() );
	Tensor<float> B0( 1, 1, b_init.begin(), b_init.end() );
	using SimpleNN = SequentialNN<Tensor<float>, MSELoss<Tensor<float>>, LinearLayer<Tensor<float>>>;
	SimpleNN nn( LinearLayer<Tensor<float>>( W0, B0 ) );
	MomentumSGDOptimizer<Tensor<float>, SimpleNN> opt( nn );
	opt.m_learning_rate  = 0.02f;
	opt.m_momentum      = 0.9f;
	opt.m_l2_regularizer = 0.0f;
	opt.m_nesterov      = true;
	opt.m_epochs         = 2;
	opt.m_batch_size     = 4;

	std::vector<Tensor<float>> inputs;
	std::vector<Tensor<float>> targets;
	for ( int r = 0; r < 4; ++r ) {
		std::vector<float> row( 2u );
		row[0] = x_h[static_cast<std::size_t>( r * 2 )];
		row[1] = x_h[static_cast<std::size_t>( r * 2 + 1 )];
		inputs.push_back( Tensor<float>( 1, 2, row.begin(), row.end() ) );
		targets.push_back( Tensor<float>(
		    1, 1, &t_h[static_cast<std::size_t>( r )], &t_h[static_cast<std::size_t>( r )] + 1 ) );
	}
	opt.train( inputs, targets );

	LinearLayer<Tensor<float>> *L = nullptr;
	nn.forEachLayer( [&]( auto &layer ) { L = &layer; } );
	REQUIRE( L != nullptr );
	require_equal_tensor( L->getWeights(), w_after, k_tol );
	require_equal_tensor( L->getBias(), b_after, k_tol );
}
