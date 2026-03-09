#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "tensor.hpp"
#include <vector>
#include <numeric>

using neural::Tensor;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

namespace {

	constexpr float eps = 1e-5f;

	Tensor<float> make_from_vec(std::size_t rows, std::size_t cols, std::vector<float> const& data) {
		return Tensor<float>(rows, cols, data.begin(), data.end());
	}
}

TEST_CASE("Tensor shape", "[tensor][shape]") {
	SECTION("default constructor gives 0x0") {
		Tensor<float> t;
		REQUIRE(t.rows() == 0u);
		REQUIRE(t.cols() == 0u);
		REQUIRE(t.size() == 0u);
	}

	SECTION("(rows, cols) constructor") {
		Tensor<float> t(2, 3);
		REQUIRE(t.rows() == 2u);
		REQUIRE(t.cols() == 3u);
		REQUIRE(t.size() == 6u);
	}

	SECTION("(rows, cols, value) initializes and shape is correct") {
		Tensor<float> t(3, 4, 1.0f);
		REQUIRE(t.rows() == 3u);
		REQUIRE(t.cols() == 4u);
		REQUIRE(t.size() == 12u);
	}

	SECTION("construction from range gives correct shape") {
		std::vector<float> data(6, 1.0f);
		Tensor<float> t(2, 3, data.begin(), data.end());
		REQUIRE(t.rows() == 2u);
		REQUIRE(t.cols() == 3u);
		REQUIRE(t.size() == 6u);
	}
}

TEST_CASE("Tensor element access and construction from data", "[tensor][access]") {
	std::vector<float> data = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
	Tensor<float> t(2, 3, data.begin(), data.end());

	REQUIRE_THAT(t(0, 0), WithinAbs(1.0f, eps));
	REQUIRE_THAT(t(0, 1), WithinAbs(2.0f, eps));
	REQUIRE_THAT(t(0, 2), WithinAbs(3.0f, eps));
	REQUIRE_THAT(t(1, 0), WithinAbs(4.0f, eps));
	REQUIRE_THAT(t(1, 1), WithinAbs(5.0f, eps));
	REQUIRE_THAT(t(1, 2), WithinAbs(6.0f, eps));
}

TEST_CASE("Tensor matmul", "[tensor][matmul]") {
	// A 2x3, B 3x2 -> C 2x2
	// A = [1 2 3; 4 5 6], B = [1 2; 3 4; 5 6]
	// C(0,0)=1+6+15=22, C(0,1)=2+8+18=28, C(1,0)=4+15+30=49, C(1,1)=8+20+36=64
	std::vector<float> a_data = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
	std::vector<float> b_data = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
	Tensor<float> A(2, 3, a_data.begin(), a_data.end());
	Tensor<float> B(3, 2, b_data.begin(), b_data.end());

	Tensor<float> C = A.matmul(B);

	REQUIRE(C.rows() == 2u);
	REQUIRE(C.cols() == 2u);
	REQUIRE_THAT(C(0, 0), WithinAbs(22.0f, eps));
	REQUIRE_THAT(C(0, 1), WithinAbs(28.0f, eps));
	REQUIRE_THAT(C(1, 0), WithinAbs(49.0f, eps));
	REQUIRE_THAT(C(1, 1), WithinAbs(64.0f, eps));
}

TEST_CASE("Tensor transpose", "[tensor][transpose]") {
	std::vector<float> data = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
	Tensor<float> A(2, 3, data.begin(), data.end());
	Tensor<float> B = A;
	B.transpose();

	REQUIRE(B.rows() == 3u);
	REQUIRE(B.cols() == 2u);
	REQUIRE_THAT(B(0, 0), WithinAbs(1.0f, eps));
	REQUIRE_THAT(B(0, 1), WithinAbs(4.0f, eps));
	REQUIRE_THAT(B(1, 0), WithinAbs(2.0f, eps));
	REQUIRE_THAT(B(1, 1), WithinAbs(5.0f, eps));
	REQUIRE_THAT(B(2, 0), WithinAbs(3.0f, eps));
	REQUIRE_THAT(B(2, 1), WithinAbs(6.0f, eps));

	SECTION("(A^T)^T equals A") {
		Tensor<float> At = A;
		At.transpose();
		At.transpose();
		REQUIRE(At.rows() == A.rows());
		REQUIRE(At.cols() == A.cols());
		for (std::size_t i = 0; i < A.rows(); ++i)
			for (std::size_t j = 0; j < A.cols(); ++j)
				REQUIRE_THAT(At(i, j), WithinAbs(A(i, j), eps));
	}
}

TEST_CASE("Tensor element-wise add", "[tensor][elementwise]") {
	std::vector<float> a_data = { 1.f, 2.f, 3.f, 4.f };
	std::vector<float> b_data = { 0.f, 1.f, 2.f, 3.f };
	Tensor<float> A(2, 2, a_data.begin(), a_data.end());
	Tensor<float> B(2, 2, b_data.begin(), b_data.end());

	Tensor<float> C = A + B;

	REQUIRE(C.rows() == 2u);
	REQUIRE(C.cols() == 2u);
	REQUIRE_THAT(C(0, 0), WithinAbs(1.0f, eps));
	REQUIRE_THAT(C(0, 1), WithinAbs(3.0f, eps));
	REQUIRE_THAT(C(1, 0), WithinAbs(5.0f, eps));  // 3 + 2
	REQUIRE_THAT(C(1, 1), WithinAbs(7.0f, eps));
}

TEST_CASE("Tensor element-wise multiply", "[tensor][elementwise]") {
	std::vector<float> a_data = { 1.f, 2.f, 3.f, 4.f };
	std::vector<float> b_data = { 2.f, 3.f, 4.f, 5.f };
	Tensor<float> A(2, 2, a_data.begin(), a_data.end());
	Tensor<float> B(2, 2, b_data.begin(), b_data.end());

	Tensor<float> C = A * B;

	REQUIRE(C.rows() == 2u);
	REQUIRE(C.cols() == 2u);
	REQUIRE_THAT(C(0, 0), WithinAbs(2.0f, eps));
	REQUIRE_THAT(C(0, 1), WithinAbs(6.0f, eps));
	REQUIRE_THAT(C(1, 0), WithinAbs(12.0f, eps));
	REQUIRE_THAT(C(1, 1), WithinAbs(20.0f, eps));
}

TEST_CASE("Tensor operator*=", "[tensor][elementwise]") {
	std::vector<float> a_data = { 1.f, 2.f, 3.f, 4.f };
	std::vector<float> b_data = { 2.f, 2.f, 2.f, 2.f };
	Tensor<float> A(2, 2, a_data.begin(), a_data.end());
	Tensor<float> B(2, 2, b_data.begin(), b_data.end());

	A *= B;

	REQUIRE_THAT(A(0, 0), WithinAbs(2.0f, eps));
	REQUIRE_THAT(A(0, 1), WithinAbs(4.0f, eps));
	REQUIRE_THAT(A(1, 0), WithinAbs(6.0f, eps));
	REQUIRE_THAT(A(1, 1), WithinAbs(8.0f, eps));
}

TEST_CASE("Tensor scalar multiply", "[tensor][scalar]") {
	std::vector<float> data = { 1.f, 2.f, 3.f, 4.f };
	Tensor<float> A(2, 2, data.begin(), data.end());

	Tensor<float> B = A * 2.0f;

	REQUIRE_THAT(B(0, 0), WithinAbs(2.0f, eps));
	REQUIRE_THAT(B(0, 1), WithinAbs(4.0f, eps));
	REQUIRE_THAT(B(1, 0), WithinAbs(6.0f, eps));
	REQUIRE_THAT(B(1, 1), WithinAbs(8.0f, eps));
}

TEST_CASE("Tensor sum", "[tensor][sum]") {
	std::vector<float> data = { 1.f, 2.f, 3.f, 4.f };
	Tensor<float> A(2, 2, data.begin(), data.end());

	REQUIRE_THAT(A.sum(), WithinAbs(10.0f, eps));
}

TEST_CASE("Tensor sum_along_axis", "[tensor][sum_along_axis]") {
	// [1 2 3; 4 5 6]
	// axis 0: sum cols -> [5 7 9]   shape (1, 3)
	// axis 1: sum rows -> [6; 15]   shape (2, 1)
	std::vector<float> data = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
	Tensor<float> A(2, 3, data.begin(), data.end());

	Tensor<float> col_sums = A.sum_along_axis(0);
	REQUIRE(col_sums.rows() == 1u);
	REQUIRE(col_sums.cols() == 3u);
	REQUIRE_THAT(col_sums(0, 0), WithinAbs(5.0f, eps));
	REQUIRE_THAT(col_sums(0, 1), WithinAbs(7.0f, eps));
	REQUIRE_THAT(col_sums(0, 2), WithinAbs(9.0f, eps));

	Tensor<float> row_sums = A.sum_along_axis(1);
	REQUIRE(row_sums.rows() == 2u);
	REQUIRE(row_sums.cols() == 1u);
	REQUIRE_THAT(row_sums(0, 0), WithinAbs(6.0f, eps));
	REQUIRE_THAT(row_sums(1, 0), WithinAbs(15.0f, eps));
}

TEST_CASE("Tensor (rows, cols) zeros", "[tensor][construction]") {
	Tensor<float> t(2, 3);
	for (std::size_t i = 0; i < t.rows(); ++i)
		for (std::size_t j = 0; j < t.cols(); ++j)
			REQUIRE_THAT(t(i, j), WithinAbs(0.0f, eps));
}

TEST_CASE("Tensor (rows, cols, value) constant", "[tensor][construction]") {
	Tensor<float> t(2, 3, 3.14f);
	for (std::size_t i = 0; i < t.rows(); ++i)
		for (std::size_t j = 0; j < t.cols(); ++j)
			REQUIRE_THAT(t(i, j), WithinAbs(3.14f, eps));
}

TEST_CASE("Tensor invalid range size throws", "[tensor][construction]") {
	std::vector<float> data = { 1.f, 2.f, 3.f };  // size 3
	REQUIRE_THROWS_AS(Tensor<float>(2, 2, data.begin(), data.end()), std::invalid_argument);
}
