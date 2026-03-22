#include "cross_entropy_softmax_loss.hpp"
#include "mse_loss.hpp"
#include "tensor.hpp"
#include <cmath>
#include "tensor_like.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

using neural::MSELoss;
using neural::SoftmaxCrossEntropyLoss;
using neural::Tensor;

static_assert( TensorLike<Tensor<float>> );

using Catch::Matchers::WithinAbs;

namespace {

constexpr float eps = 1e-5f;

} // namespace

TEST_CASE( "MSELoss forward value when pred equals target", "[mse_loss][forward][numerical]" )
{
	MSELoss<Tensor<float>> loss;

	std::vector<float> data = { 1.f, 2.f, 3.f, 4.f };
	Tensor<float> pred( 2, 2, data.begin(), data.end() );
	Tensor<float> target( 2, 2, data.begin(), data.end() );

	auto const L = loss.forward( pred, target );

	REQUIRE_THAT( L, WithinAbs( 0.0f, eps ) );
}

TEST_CASE( "MSELoss forward value for known difference", "[mse_loss][forward][numerical]" )
{
	MSELoss<Tensor<float>> loss;

	// pred = [1, 2], target = [3, 4]
	// diff = [-2, -2], diff^2 = [4, 4], mean = 4
	std::vector<float> pred_data = { 1.f, 2.f };
	std::vector<float> target_data = { 3.f, 4.f };
	Tensor<float> pred( 1, 2, pred_data.begin(), pred_data.end() );
	Tensor<float> target( 1, 2, target_data.begin(), target_data.end() );

	auto const L = loss.forward( pred, target );

	REQUIRE_THAT( L, WithinAbs( 4.0f, eps ) );
}

TEST_CASE( "MSELoss forward value for batch", "[mse_loss][forward][numerical]" )
{
	MSELoss<Tensor<float>> loss;

	// pred = [1, 2; 3, 4], target = [0, 0; 0, 0]
	// diff = pred, diff^2 = [1, 4; 9, 16], sum = 30, mean = 30/4 = 7.5
	std::vector<float> pred_data = { 1.f, 2.f, 3.f, 4.f };
	std::vector<float> target_data = { 0.f, 0.f, 0.f, 0.f };
	Tensor<float> pred( 2, 2, pred_data.begin(), pred_data.end() );
	Tensor<float> target( 2, 2, target_data.begin(), target_data.end() );

	auto const L = loss.forward( pred, target );

	REQUIRE_THAT( L, WithinAbs( 7.5f, eps ) );
}

TEST_CASE( "MSELoss backward gradient shape", "[mse_loss][backward][shape]" )
{
	MSELoss<Tensor<float>> loss;

	Tensor<float> pred( 2, 5, 1.0f );
	Tensor<float> target( 2, 5, 0.0f );
	loss.forward( pred, target );

	Tensor<float> grad = loss.backward();

	REQUIRE( grad.rows() == 2u );
	REQUIRE( grad.cols() == 5u );
}

TEST_CASE( "MSELoss backward gradient when pred equals target", "[mse_loss][backward][numerical]" )
{
	MSELoss<Tensor<float>> loss;

	std::vector<float> data = { 1.f, 2.f, 3.f };
	Tensor<float> pred( 1, 3, data.begin(), data.end() );
	Tensor<float> target( 1, 3, data.begin(), data.end() );
	loss.forward( pred, target );

	Tensor<float> grad = loss.backward();

	// diff = 0, so grad = (2/n) * 0 = 0
	REQUIRE_THAT( grad( 0, 0 ), WithinAbs( 0.0f, eps ) );
	REQUIRE_THAT( grad( 0, 1 ), WithinAbs( 0.0f, eps ) );
	REQUIRE_THAT( grad( 0, 2 ), WithinAbs( 0.0f, eps ) );
}

TEST_CASE( "MSELoss backward gradient formula", "[mse_loss][backward][numerical]" )
{
	MSELoss<Tensor<float>> loss;

	// pred = [1, 2], target = [3, 4], n = 2
	// diff = [-2, -2]
	// dL/dpred = (2/n) * diff = (2/2) * [-2, -2] = [-2, -2]
	std::vector<float> pred_data = { 1.f, 2.f };
	std::vector<float> target_data = { 3.f, 4.f };
	Tensor<float> pred( 1, 2, pred_data.begin(), pred_data.end() );
	Tensor<float> target( 1, 2, target_data.begin(), target_data.end() );
	loss.forward( pred, target );

	Tensor<float> grad = loss.backward();

	REQUIRE_THAT( grad( 0, 0 ), WithinAbs( -2.0f, eps ) );
	REQUIRE_THAT( grad( 0, 1 ), WithinAbs( -2.0f, eps ) );
}

TEST_CASE( "MSELoss forward throws on shape mismatch", "[mse_loss][forward][validation]" )
{
	MSELoss<Tensor<float>> loss;

	Tensor<float> pred( 2, 3, 1.0f );
	Tensor<float> target( 2, 2, 1.0f );

	REQUIRE_THROWS_AS( loss.forward( pred, target ), std::invalid_argument );
}

TEST_CASE( "SoftmaxCrossEntropyLoss forward uniform logits one-hot class 0", "[softmax_ce][forward][numerical]" )
{
	SoftmaxCrossEntropyLoss<Tensor<float>> loss;

	// 3 classes, logits all 0 -> p = (1/3,1/3,1/3), CE = -log(1/3)
	std::vector<float> logits_data = { 0.f, 0.f, 0.f };
	std::vector<float> target_data = { 1.f, 0.f, 0.f };
	Tensor<float> logits( 1, 3, logits_data.begin(), logits_data.end() );
	Tensor<float> target( 1, 3, target_data.begin(), target_data.end() );

	auto const L = loss.forward( logits, target );
	float const expected = std::log( 3.0f );

	REQUIRE_THAT( L, WithinAbs( expected, 1e-4f ) );
}

TEST_CASE( "SoftmaxCrossEntropyLoss backward matches p minus t scaled", "[softmax_ce][backward][numerical]" )
{
	SoftmaxCrossEntropyLoss<Tensor<float>> loss;

	std::vector<float> logits_data = { 0.f, 0.f, 0.f };
	std::vector<float> target_data = { 1.f, 0.f, 0.f };
	Tensor<float> logits( 1, 3, logits_data.begin(), logits_data.end() );
	Tensor<float> target( 1, 3, target_data.begin(), target_data.end() );

	loss.forward( logits, target );
	Tensor<float> grad = loss.backward();

	float const p = 1.0f / 3.0f;
	REQUIRE_THAT( grad( 0, 0 ), WithinAbs( p - 1.0f, eps ) );
	REQUIRE_THAT( grad( 0, 1 ), WithinAbs( p - 0.0f, eps ) );
	REQUIRE_THAT( grad( 0, 2 ), WithinAbs( p - 0.0f, eps ) );
}

TEST_CASE( "SoftmaxCrossEntropyLoss batch mean over two rows", "[softmax_ce][forward][batch]" )
{
	SoftmaxCrossEntropyLoss<Tensor<float>> loss;

	std::vector<float> logits_data = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
	std::vector<float> target_data = { 1.f, 0.f, 0.f, 0.f, 1.f, 0.f };
	Tensor<float> logits( 2, 3, logits_data.begin(), logits_data.end() );
	Tensor<float> target( 2, 3, target_data.begin(), target_data.end() );

	auto const L = loss.forward( logits, target );
	float const row_ce = std::log( 3.0f );

	REQUIRE_THAT( L, WithinAbs( row_ce, 1e-4f ) );
}

TEST_CASE( "SoftmaxCrossEntropyLoss forward throws on shape mismatch", "[softmax_ce][forward][validation]" )
{
	SoftmaxCrossEntropyLoss<Tensor<float>> loss;

	Tensor<float> logits( 2, 3, 1.0f );
	Tensor<float> target( 2, 2, 1.0f );

	REQUIRE_THROWS_AS( loss.forward( logits, target ), std::invalid_argument );
}
