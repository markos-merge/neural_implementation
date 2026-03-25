#ifndef TENSOR_LIKE_HPP
#define TENSOR_LIKE_HPP

#include <concepts>
#include <cstddef>
#include <utility>

template < typename MatrixType >
concept TensorLike = requires( MatrixType m ) {
	typename MatrixType::value_type;
	typename MatrixType::matrix_type;
	typename MatrixType::size_type;

	{ m.rows() } -> std::convertible_to<std::size_t>;
	{ m.cols() } -> std::convertible_to<std::size_t>;
	{ m.size() } -> std::convertible_to<std::size_t>;
	{
		m.operator()( 0, 0 )
	} -> std::convertible_to<typename MatrixType::value_type>;
	{
		std::as_const( m ).operator()( 0, 0 )
	} -> std::convertible_to<typename MatrixType::value_type const>;

	{
		m.assign( static_cast<typename MatrixType::value_type>( 0. ) )
	};

	{
		m.transpose()
	} -> std::convertible_to<MatrixType>;
	
	{
		m.transposeInPlace()
	} -> std::convertible_to<MatrixType>;

	{
		m.reshape( m.rows(), m.cols() )
	} -> std::convertible_to<MatrixType>;
	{
		m.reshapeInPlace( m.rows(), m.cols() )
	} -> std::convertible_to<MatrixType>;
	
	{
		m.matmul( m )
	} -> std::convertible_to<MatrixType>;

	{
		m.matmulInPlace( m )
	} -> std::convertible_to<MatrixType>;

	{
		m.operator+( m )
	} -> std::convertible_to<MatrixType>;
	
	{
		m.operator+=( m )
	} -> std::convertible_to<MatrixType>;

	{
		m.operator*( m )
	} -> std::convertible_to<MatrixType>;
	{
		m.operator*=( m )
	} -> std::convertible_to<MatrixType>;
	{
		m.operator*( static_cast<typename MatrixType::value_type>(0.) )
	} -> std::convertible_to<MatrixType>;
	{
		m.operator*=( static_cast<typename MatrixType::value_type>(0.) )
	} -> std::convertible_to<MatrixType>;

	{
		m.cwiseGreater( static_cast<typename MatrixType::value_type>(0.) )
	} -> std::convertible_to<MatrixType>;
	{
		m.cwiseOneMinus()
	} -> std::convertible_to<MatrixType>;

	{
		m.sum()
	} -> std::convertible_to<typename MatrixType::value_type>;

	{
		m.asum()
	} -> std::convertible_to<typename MatrixType::value_type>;

	{
		m.sum_along_axis( 0 )
	} -> std::convertible_to<MatrixType>;
};
#endif