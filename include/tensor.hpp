#ifndef NEURAL_IMPL_TENSOR_CPP
#define NEURAL_IMPL_TENSOR_CPP

#include <Eigen/Dense>
#include <cmath>
#include <iterator>
#include <random>

namespace neural {

template <typename T = float>
class Tensor
{
	public:
		using value_type = T;
		using matrix_type = Eigen::Matrix<value_type, Eigen::Dynamic,
		                                  Eigen::Dynamic, Eigen::RowMajor>;
		using size_type = std::size_t;

	public:
		Tensor() noexcept;
		Tensor( std::size_t rows, std::size_t cols ) noexcept;
		template <std::random_access_iterator It>
		Tensor( std::size_t rows, std::size_t cols, It begin, It end );
		Tensor( std::size_t rows, std::size_t cols, value_type value ) noexcept;
		explicit Tensor( matrix_type mat ) noexcept;
		Tensor( Tensor const &other ) noexcept;
		Tensor( Tensor &&other ) noexcept;
		Tensor &operator=( Tensor const &other ) noexcept;
		Tensor &operator=( Tensor &&other ) noexcept;
		~Tensor() = default;
		std::size_t rows() const;
		std::size_t cols() const;
		std::size_t size() const;

		value_type operator()( std::size_t row, std::size_t col );
		value_type const operator()( std::size_t row, std::size_t col ) const;
	
		void assign( std::size_t row, std::size_t col, value_type value );

		/// Copy \p other 's first row into row \p row of this tensor (same column count).
		void assignTensorAsRow( std::size_t row, Tensor const &other );

		Tensor<T> transpose() const;
		Tensor<T> &transposeInPlace() noexcept;
		Tensor<T> reshape( std::size_t rows, std::size_t cols ) const;
		Tensor<T> &reshapeInPlace( std::size_t rows, std::size_t cols ) noexcept;

		Tensor matmul( Tensor const &other ) const;
		Tensor &matmulInPlace( Tensor const &other );
	
		Tensor divideRowsWithCol( Tensor const &other ) const;
		Tensor &divideRowsWithColInPlace( Tensor const &other );

		/// Largest absolute value among all elements.
		value_type maxCoeff() const;

		Tensor operator+( Tensor const &other ) const;
		Tensor &operator+=( Tensor const &other );
		Tensor operator-( Tensor const &other ) const;
		Tensor &operator-=( Tensor const &other );
		Tensor addColwise( Tensor const &col ) const;
		Tensor &addColwiseInPlace( Tensor const &col );
		/// For each row i, subtract col(i,0) from all entries in that row. col has shape
		/// (rows(), 1).
		Tensor subtractColwise( Tensor const &col ) const;
		Tensor operator*( Tensor const &other ) const; // element-wise
		Tensor &operator*=( Tensor const &other );
		Tensor operator*( value_type scalar ) const;
		Tensor &operator*=( value_type scalar );

		Tensor cwiseGreater( value_type scalar ) const;
		Tensor cwiseOneMinus() const;
		Tensor cwiseSigmoid() const;
		Tensor cwiseExp() const;
		Tensor cwiseLog() const;

		value_type sum() const;
		Tensor sum_along_axis( std::size_t axis ) const;
		Tensor max_along_axis( std::size_t axis ) const;

		template <typename Generator>
		void randomize( Generator &generator ) noexcept;
		void randomize() noexcept;
		/// He initialization for ReLU: uniform(-sqrt(2/fan_in), sqrt(2/fan_in))
		void randomizeHe( std::size_t fan_in ) noexcept;
	private:
		matrix_type m_mat;
};

/***************** Implementation  **********************/
template <typename T>
Tensor<T>::Tensor() noexcept : m_mat( 0, 0 )
{
}

template <typename T>
Tensor<T>::Tensor( std::size_t rows, std::size_t cols ) noexcept
    : m_mat( rows, cols )
{
	m_mat.setZero();
}

template <typename T>
template <std::random_access_iterator It>
Tensor<T>::Tensor( std::size_t rows, std::size_t cols, It begin, It end )
    : m_mat( rows, cols )
{
	std::size_t const size = std::distance( begin, end );

	if ( size != rows * cols ) {
		throw std::invalid_argument(
		    "The number of elements in the range must be equal to the number of "
		    "elements in the tensor" );
	}

	std::copy( begin, end, m_mat.data() );
}

template <typename T>
Tensor<T>::Tensor( std::size_t rows, std::size_t cols,
                   value_type value ) noexcept
    : m_mat( rows, cols )
{
	m_mat.setConstant( value );
}

template <typename T>
Tensor<T>::Tensor( matrix_type mat ) noexcept : m_mat( std::move( mat ) )
{
}

template <typename T>
Tensor<T>::Tensor( Tensor const &other ) noexcept : m_mat( other.m_mat )
{
}

template <typename T>
Tensor<T>::Tensor( Tensor &&other ) noexcept : m_mat( std::move( other.m_mat ) )
{
}

template <typename T>
Tensor<T> &Tensor<T>::operator=( Tensor const &other ) noexcept
{
	m_mat = other.m_mat;
	return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::operator=( Tensor &&other ) noexcept
{
	m_mat = std::move( other.m_mat );
	return *this;
}

template <typename T>
std::size_t Tensor<T>::rows() const
{
	return m_mat.rows();
}

template <typename T>
std::size_t Tensor<T>::cols() const
{
	return m_mat.cols();
}

template <typename T>
std::size_t Tensor<T>::size() const
{
	return m_mat.size();
}

template <typename T>
typename Tensor<T>::value_type Tensor<T>::operator()( std::size_t row,
                                                      std::size_t col )
{
	return m_mat( row, col );
}

template <typename T>
typename Tensor<T>::value_type const
Tensor<T>::operator()( std::size_t row, std::size_t col ) const
{
	return m_mat( row, col );
}

template <typename T>
void Tensor<T>::assign( std::size_t row, std::size_t col, value_type value )
{
	m_mat( row, col ) = value;
}

template <typename T>
void Tensor<T>::assignTensorAsRow( std::size_t row, Tensor const &other )
{
	if ( row >= static_cast<std::size_t>( m_mat.rows() ) ) {
		throw std::out_of_range( "Tensor::assignTensorAsRow(): row out of range" );
	}
	if ( other.m_mat.rows() < 1
	     || static_cast<std::size_t>( other.m_mat.cols() ) != static_cast<std::size_t>( m_mat.cols() ) ) {
		throw std::invalid_argument(
		    "Tensor::assignTensorAsRow(): other must have rows>=1 and same cols()" );
	}
	m_mat.row( static_cast<Eigen::Index>( row ) ) = other.m_mat.row( 0 );
}

template <typename T>
Tensor<T> Tensor<T>::transpose() const
{
	Tensor<T> result( *this );

	result.transposeInPlace();

	return result;
}

template <typename T>
Tensor<T> &Tensor<T>::transposeInPlace() noexcept
{
	m_mat = m_mat.transpose().eval();
	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::reshape( std::size_t rows, std::size_t cols ) const
{
	return Tensor<T>( m_mat.template reshaped<Eigen::RowMajor>( rows, cols ) );
}

template <typename T>
Tensor<T> &Tensor<T>::reshapeInPlace( std::size_t rows,
                                      std::size_t cols ) noexcept
{
	m_mat = m_mat.template reshaped<Eigen::RowMajor>( rows, cols );
	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::matmul( Tensor const &other ) const
{
	return Tensor<T>( ( m_mat * other.m_mat ).eval() );
}

template <typename T>
Tensor<T> &Tensor<T>::matmulInPlace( Tensor const &other )
{
	m_mat = ( m_mat * other.m_mat ).eval();
	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::divideRowsWithCol( Tensor const &other ) const
{
	return Tensor<T>(
	    ( m_mat.array().colwise() / other.m_mat.col( 0 ).array() ).matrix().eval() );
}

template <typename T>
Tensor<T> &Tensor<T>::divideRowsWithColInPlace( Tensor const &other )
{
	m_mat.array().colwise() /= other.m_mat.col( 0 ).array();
	return *this;
}

template <typename T>
typename Tensor<T>::value_type Tensor<T>::maxCoeff() const
{
	if ( m_mat.size() == 0 ) {
		return value_type{};
	}
	return m_mat.cwiseAbs().maxCoeff();
}

template <typename T>
Tensor<T> Tensor<T>::operator+( Tensor const &other ) const
{
	return Tensor<T>( ( m_mat + other.m_mat ).eval() );
}

template <typename T>
Tensor<T> &Tensor<T>::operator+=( Tensor const &other )
{
	m_mat += other.m_mat;
	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::operator-( Tensor const &other ) const
{
	return Tensor<T>( ( m_mat - other.m_mat ).eval() );
}

template <typename T>
Tensor<T> &Tensor<T>::operator-=( Tensor const &other )
{
	m_mat -= other.m_mat;
	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::addColwise( Tensor const &col ) const
{
	// col is (cols(), 1); broadcast bias across batch rows (Eigen array + does not broadcast).
	return Tensor<T>(
	    ( m_mat.rowwise() + col.m_mat.col( 0 ).transpose() ).eval() );
}

template <typename T>
Tensor<T> &Tensor<T>::addColwiseInPlace( Tensor const &col )
{
	m_mat.rowwise() += col.m_mat.col( 0 ).transpose();
	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::subtractColwise( Tensor const &col ) const
{
	return Tensor<T>(
	    ( m_mat.array().colwise() - col.m_mat.col( 0 ).array() ).matrix().eval() );
}

template <typename T>
Tensor<T> Tensor<T>::operator*( Tensor const &other ) const
{
	return Tensor<T>( ( m_mat.array() * other.m_mat.array() ).matrix().eval() );
}

template <typename T>
Tensor<T> &Tensor<T>::operator*=( Tensor const &other )
{
	m_mat.array() *= other.m_mat.array();
	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::operator*( value_type scalar ) const
{
	return Tensor<T>( ( m_mat * scalar ).eval() );
}

template <typename T>
Tensor<T> &Tensor<T>::operator*=( value_type scalar )
{
	m_mat *= scalar;
	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::cwiseGreater( value_type scalar ) const
{
	return Tensor<T>( ( m_mat.array() > scalar ).template cast<value_type>().matrix().eval() );
}

template <typename T>
Tensor<T> Tensor<T>::cwiseOneMinus() const
{
	return Tensor<T>( ( 1.0 - m_mat.array() ).matrix().eval() );
}

template <typename T>
Tensor<T> Tensor<T>::cwiseSigmoid() const
{
	return Tensor<T>( ( 1.0 / ( 1.0 + (-m_mat.array()).exp() ) ).matrix().eval() );
}

template <typename T>
Tensor<T> Tensor<T>::cwiseExp() const
{
	return Tensor<T>( ( m_mat.array().exp() ).matrix().eval() );
}

template <typename T>
Tensor<T> Tensor<T>::cwiseLog() const
{
	return Tensor<T>( ( m_mat.array().log() ).matrix().eval() );
}

template <typename T>
T Tensor<T>::sum() const
{
	return m_mat.sum();
}

template <typename T>
Tensor<T> Tensor<T>::sum_along_axis( std::size_t axis ) const
{
	if ( axis == 0 ) {
		return Tensor<T>( m_mat.colwise().sum() );
	}
	return Tensor<T>( m_mat.rowwise().sum() );
}

template <typename T>
Tensor<T> Tensor<T>::max_along_axis( std::size_t axis ) const
{
	if ( axis == 0 ) {
		return Tensor<T>( m_mat.colwise().maxCoeff() );
	}
	return Tensor<T>( m_mat.rowwise().maxCoeff() );
}

template <typename T>
template <typename Generator>
void Tensor<T>::randomize( Generator &generator ) noexcept
{
	std::generate( m_mat.data(), m_mat.data() + m_mat.size(), generator );
}

template <typename T>
void Tensor<T>::randomize() noexcept
{
	std::random_device rd;
	std::mt19937 gen( rd() );
	// std::uniform_real_distribution is only defined for float/double/long double;
	// Eigen::half and other custom scalars use double + cast.
	std::uniform_real_distribution<double> dis( 0.0, 1.0 );
	auto generator = [&]() { return static_cast<T>( dis( gen ) ); };
	randomize( generator );
}

template <typename T>
void Tensor<T>::randomizeHe( std::size_t fan_in ) noexcept
{
	double const scale = std::sqrt( 2.0 / static_cast<double>( fan_in ) );
	std::random_device rd;
	std::mt19937 gen( rd() );
	std::uniform_real_distribution<double> dis( -scale, scale );
	auto generator = [&]() { return static_cast<T>( dis( gen ) ); };
	randomize( generator );
}
} // namespace neural

#endif // NEURAL_IMPL_TENSOR_CPP