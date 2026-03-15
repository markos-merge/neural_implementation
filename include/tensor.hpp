#ifndef NEURAL_IMPL_TENSOR_CPP
#define NEURAL_IMPL_TENSOR_CPP

#include <Eigen/Dense>
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

		Tensor<T> transpose() const;
		Tensor<T> &transposeInPlace() noexcept;
		Tensor<T> reshape( std::size_t rows, std::size_t cols ) const;
		Tensor<T> &reshapeInPlace( std::size_t rows, std::size_t cols ) noexcept;

		Tensor matmul( Tensor const &other ) const;
		Tensor &matmulInPlace( Tensor const &other );

		Tensor operator+( Tensor const &other ) const;
		Tensor &operator+=( Tensor const &other );
		Tensor &addColwise( Tensor const &col );
		Tensor operator*( Tensor const &other ) const; // element-wise
		Tensor &operator*=( Tensor const &other );
		Tensor operator*( value_type scalar ) const;
		Tensor &operator*=( value_type scalar );

		Tensor cwiseGreater( value_type scalar ) const;
		Tensor cwiseOneMinus() const;
		Tensor cwiseSigmoid() const;

		template <typename UnaryOperator>
		Tensor cwiseApply( UnaryOperator unary_operator ) const;

		value_type sum() const;
		Tensor sum_along_axis( std::size_t axis ) const;

		template <typename Generator>
		void randomize( Generator &generator ) noexcept;
		void randomize() noexcept;
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
Tensor<T> &Tensor<T>::addColwise( Tensor const &col )
{
	// col is (O, 1), this is (B, O). Add col to each row: broadcast (1,O) over rows.
	m_mat = ( m_mat + col.m_mat.transpose().replicate( rows(), 1 ) ).eval();
	return *this;
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
T Tensor<T>::sum() const
{
	return m_mat.sum();
}

template <typename T>
Tensor<T> Tensor<T>::sum_along_axis( std::size_t axis ) const
{
	Tensor<T> result( axis == 0 ? 1 : rows(), axis == 0 ? cols() : 1 );
	if ( axis == 0 ) {
		result.m_mat = m_mat.colwise().sum();
	} else {
		result.m_mat = m_mat.rowwise().sum();
	}

	return result;
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
	std::uniform_real_distribution<T> dis( 0.0, 1.0 );
	auto generator = [&]() { return dis( gen ); };
	randomize( generator );
}
} // namespace neural

#endif // NEURAL_IMPL_TENSOR_CPP