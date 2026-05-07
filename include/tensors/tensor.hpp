#ifndef NEURAL_IMPL_TENSOR_CPP
#define NEURAL_IMPL_TENSOR_CPP

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#include <type_traits>

namespace neural {

template <typename U>
struct is_cuda_tensor : std::false_type
{
};
template <typename U>
inline constexpr bool is_cuda_tensor_v = is_cuda_tensor<U>::value;

template <typename T = float>
class Tensor
{
	public:
		using value_type = T;
		using matrix_type = Eigen::Matrix<value_type, Eigen::Dynamic,
		                                  Eigen::Dynamic, Eigen::RowMajor>;
		using size_type = std::size_t;

		template <typename type >
		using tensor_alias = Tensor<type>;

	public:
		Tensor() noexcept;
		Tensor( std::size_t rows, std::size_t cols ) noexcept;
		Tensor( std::array<std::size_t, 2> shape ) noexcept;
		template <std::random_access_iterator It>
		Tensor( std::size_t rows, std::size_t cols, It begin, It end );
		Tensor( std::size_t rows, std::size_t cols, value_type value ) noexcept;
		explicit Tensor( matrix_type mat ) noexcept;
		Tensor( Tensor const &other ) noexcept;
		Tensor( Tensor &&other ) noexcept;//todo: use m_own
		Tensor &operator=( Tensor const &other );
		Tensor &operator=( Tensor &&other );

		Tensor( T *data, std::size_t rows, std::size_t cols, bool own ) noexcept;

		~Tensor();
		std::size_t rows() const;
		std::size_t cols() const;
		std::size_t size() const;
		std::array<std::size_t, 2> shape() const;
		value_type *data() noexcept;
		value_type const *data() const noexcept;

		value_type operator()( std::size_t row, std::size_t col );
		value_type const operator()( std::size_t row, std::size_t col ) const;
	
		void assign( std::size_t row, std::size_t col, value_type value );
		void assign( value_type value );
		void assign( value_type const *data, std::size_t size );

		void assignTensorAsRow( std::size_t row, Tensor const &other );
		void assignTensor( value_type *value, std::size_t size );

		Tensor<T> transpose() const;
		Tensor<T> &transposeInPlace() noexcept;
		Tensor<T> reshape( std::size_t rows, std::size_t cols ) const;
		Tensor<T> &reshapeInPlace( std::size_t rows, std::size_t cols );

		Tensor matmul( Tensor const &other ) const;
		Tensor &matmulInPlace( Tensor const &other );

		Tensor &matmulInPlace( Tensor const &m1, Tensor const &m2, bool transpose_first = false,
		                       bool transpose_second = false );

		Tensor divideRowsWithCol( Tensor const &other ) const;
		Tensor &divideRowsWithColInPlace( Tensor const &other );
		Tensor &mulNSubstractInPlace( Tensor const &other, Tensor::value_type scalar );
		/// BLAS-style \c axpy: \c *this := alpha * x + *this (same shape as \p x).
		Tensor &axpy( Tensor const &x, Tensor::value_type alpha );
		/// BLAS-style \c geam (in-place): \c *this := alpha * *this + beta * B.
		Tensor &geam( value_type alpha, value_type beta, Tensor const &B );
		/// Elementwise product into \p out; resizes \p out if its shape differs from \c *this.
		void elementwiseMultiply( Tensor const &other, Tensor &out ) const;

		value_type maxCoeff() const;

		Tensor operator+( Tensor const &other ) const;
		Tensor &operator+=( Tensor const &other );
		Tensor operator-( Tensor const &other ) const;
		Tensor &operator-=( Tensor const &other );
		Tensor addColwise( Tensor const &col ) const;
		Tensor &addColwiseInPlace( Tensor const &col );
		Tensor subtractColwise( Tensor const &col ) const;
		Tensor operator*( Tensor const &other ) const;
		Tensor &operator*=( Tensor const &other );
		Tensor operator*( value_type scalar ) const;
		Tensor &operator*=( value_type scalar );

		Tensor cwiseGreater( value_type scalar ) const;
		Tensor &cwiseGreaterInPlace( Tensor const &other, value_type scalar );
		Tensor cwiseOneMinus() const;
		void cwiseSigmoid( Tensor &out ) const;
		Tensor cwiseSigmoid() const;
		Tensor cwiseExp() const;
		Tensor cwiseLog() const;

		value_type sum() const;
		value_type asum() const;
		Tensor sumAlongAxis( std::size_t axis, bool transpose_out = false ) const;
		Tensor sum_along_axis( std::size_t axis ) const { return sumAlongAxis( axis, false ); }

		Tensor &sumAlongAxisInPlace( Tensor const &src, std::size_t axis, bool transpose_out = false );

		Tensor max_along_axis( std::size_t axis ) const;

		Tensor<Eigen::Index> argmaxAlongAxis( std::size_t axis ) const;

		template <typename Generator>
		void randomize( Generator &generator ) noexcept;
		/// Seeded uniform \f$[0, 1)\f$; deterministic given \p seed. Useful for reproducible
		/// consumers (e.g. \c DropoutLayer driving a tensor RNG from its own seeded state).
		void randomize( std::uint64_t seed ) noexcept;
		void randomizeHe( std::size_t fan_in, std::uint64_t seed ) noexcept;
		/// PyTorch \c nn.Linear / \c nn.Conv default: \c kaiming_uniform_(\,a=√5\,) → uniform
		/// in \f$(-1/\sqrt{\text{fan\_in}},\,1/\sqrt{\text{fan\_in}})\f$ (see PyTorch \c reset_parameters).
		void randomizePytorchDefault( std::size_t fan_in, std::uint64_t seed ) noexcept;
	private:
		Eigen::Map<matrix_type> map_ref() noexcept
		{
			return Eigen::Map<matrix_type>( m_data, static_cast<Eigen::Index>( m_rows ),
			                                static_cast<Eigen::Index>( m_cols ) );
		}
		Eigen::Map<const matrix_type> map_ref() const noexcept
		{
			return Eigen::Map<const matrix_type>( m_data, static_cast<Eigen::Index>( m_rows ),
			                                      static_cast<Eigen::Index>( m_cols ) );
		}

		/// Non-owning tensors borrow storage; resizing or replacing \c m_data is not allowed.
		void throw_if_non_owning_realloc( char const *context ) const
		{
			if ( !m_own ) {
				throw std::invalid_argument( context );
			}
		}

		T *m_data = nullptr;
		std::size_t m_rows = 0;
		std::size_t m_cols = 0;

		bool m_own = true;
};

template <typename T>
Tensor<T>::Tensor() noexcept :
m_own( true ),
m_data( nullptr ),
m_rows( 0 ),
m_cols( 0 )
{
}

template <typename T>
Tensor<T>::Tensor( std::size_t rows, std::size_t cols ) noexcept:
m_own( true ),
m_data( new T[rows * cols] ),
m_rows( rows ),
m_cols( cols )
{
	std::fill( m_data, m_data + rows * cols, 0 );
}

template <typename T>
Tensor<T>::Tensor( std::array<std::size_t, 2> shape ) noexcept:
m_own( true ),
m_data( new T[shape[0] * shape[1]] ),
m_rows( shape[0] ),
m_cols( shape[1] )
{
	std::fill( m_data, m_data + shape[0] * shape[1], 0 );
}

template <typename T>
template <std::random_access_iterator It>
Tensor<T>::Tensor( std::size_t rows, std::size_t cols, It begin, It end ):
m_own( true ),
m_data( new T[rows * cols] ),
m_rows( rows ),
m_cols( cols )
{
	std::size_t const size = std::distance( begin, end );

	if ( size != rows * cols ) {
		throw std::invalid_argument(
		    "The number of elements in the range must be equal to the number of "
		    "elements in the tensor" );
	}

	std::copy( begin, end, m_data );
}

template <typename T>
Tensor<T>::Tensor( std::size_t rows, std::size_t cols,
                   value_type value ) noexcept:
m_own( true ),
m_data( new T[rows * cols] ),
m_rows( rows ),
m_cols( cols )
{
	std::fill( m_data, m_data + rows * cols, value );
}

template <typename T>
Tensor<T>::Tensor( matrix_type mat ) noexcept :
m_own( true ),
m_data( new T[mat.rows() * mat.cols()] ),
m_rows( mat.rows() ),
m_cols( mat.cols() )
{
	std::copy( mat.data(), mat.data() + mat.rows() * mat.cols(), m_data );
}

template <typename T>
Tensor<T>::Tensor( Tensor const &other ) noexcept :
m_own( true ),
m_data( new T[other.m_rows * other.m_cols] ),
m_rows( other.m_rows ),
m_cols( other.m_cols )
{
	if ( m_rows * m_cols > 0u ) {
		std::copy( other.m_data, other.m_data + m_rows * m_cols, m_data );
	}
}

template <typename T>
Tensor<T>::Tensor( T *data, std::size_t rows, std::size_t cols, bool own ) noexcept :
m_own( own ),
m_data( data ),
m_rows( rows ),
m_cols( cols )
{
}

template <typename T>
Tensor<T>::Tensor( Tensor &&other ) noexcept :
m_own( other.m_own ),
m_data( other.m_data ),
m_rows( other.m_rows ),
m_cols( other.m_cols )
{
	other.m_data = nullptr;
	other.m_rows = 0;
	other.m_cols = 0;
}

template <typename T>
Tensor<T> &Tensor<T>::operator=( Tensor const &other )
{
	if ( m_own ) {
		if ( this->size() != other.size() ) {
			delete[] m_data;
			m_data = new T[other.m_rows * other.m_cols];
		}
		std::copy( other.m_data, other.m_data + other.m_rows * other.m_cols, m_data );
		m_rows = other.m_rows;
		m_cols = other.m_cols;
		return *this;
	}
	if ( m_rows != other.m_rows || m_cols != other.m_cols ) {
		throw std::invalid_argument(
		    "Tensor::operator=(Tensor const &): non-owning tensor requires matching shape" );
	}
	std::copy( other.m_data, other.m_data + size(), m_data );
	return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::operator=( Tensor &&other )
{
	if ( this == &other ) {
		return *this;
	}
	throw_if_non_owning_realloc(
	    "Tensor::operator=(Tensor &&): cannot move-assign into a non-owning tensor" );
	if ( m_own ) {
		delete[] m_data;
	}
	m_data = other.m_data;
	m_rows = other.m_rows;
	m_cols = other.m_cols;
	m_own = other.m_own;
	other.m_data = nullptr;
	other.m_rows = 0;
	other.m_cols = 0;
	other.m_own = true;
	return *this;
}

template <typename T>
Tensor<T>::~Tensor()
{
	if ( m_own ) {
		delete[] m_data;
	}
	m_data = nullptr;
	m_rows = 0;
	m_cols = 0;
}

template <typename T>
std::size_t Tensor<T>::rows() const
{
	return m_rows;
}

template <typename T>
std::size_t Tensor<T>::cols() const
{
	return m_cols;
}

template <typename T>
std::size_t Tensor<T>::size() const
{
	return m_rows * m_cols;
}

template <typename T>
std::array<std::size_t, 2> Tensor<T>::shape() const
{
	return { rows(), cols() };
}

template <typename T>
T *Tensor<T>::data() noexcept
{
	return m_data;
}

template <typename T>
T const *Tensor<T>::data() const noexcept
{
	return m_data;
}

template <typename T>
typename Tensor<T>::value_type Tensor<T>::operator()( std::size_t row,
                                                      std::size_t col )
{
	return m_data[row * m_cols + col];
}

template <typename T>
typename Tensor<T>::value_type const
Tensor<T>::operator()( std::size_t row, std::size_t col ) const
{
	return m_data[row * m_cols + col];
}

template <typename T>
void Tensor<T>::assign( std::size_t row, std::size_t col, value_type value )
{
	m_data[row * m_cols + col] = value;
}

template <typename T>
void Tensor<T>::assign( value_type value )
{
	std::fill( m_data, m_data + size(), value );
}

template <typename T>
void Tensor<T>::assignTensorAsRow( std::size_t row, Tensor const &other )
{
	if ( row >= static_cast<std::size_t>( m_rows ) ) {
		throw std::out_of_range( "Tensor::assignTensorAsRow(): row out of range" );
	}
	if ( other.m_rows < 1 ) {
		throw std::invalid_argument(
		    "Tensor::assignTensorAsRow(): other must have rows>=1" );
	}
	std::size_t const dest_cols = static_cast<std::size_t>( m_cols );
	std::size_t const other_cols = static_cast<std::size_t>( other.m_cols );
	if ( other_cols == dest_cols ) {
		Eigen::Map<matrix_type> mapped_pointer( m_data, rows(), cols() );
		Eigen::Map<matrix_type> other_mapped_pointer( other.m_data, other.rows(), other.cols() );
		mapped_pointer.row( static_cast<Eigen::Index>( row ) ) = other_mapped_pointer.row( 0 );
	} else if ( other.size() == dest_cols ) {
		Eigen::Map<const matrix_type> const as_row( other.m_data, 1, static_cast<Eigen::Index>( dest_cols ) );
		Eigen::Map<matrix_type> mapped_pointer( m_data, rows(), cols() );
		mapped_pointer.row( static_cast<Eigen::Index>( row ) ) = as_row;
	} else {
		throw std::invalid_argument(
		    "Tensor::assignTensorAsRow(): other.cols() or other.size() must match destination cols" );
	}
}

template < typename T >
void Tensor<T>::assignTensor( value_type *value, std::size_t size )
{
	if ( size != this->size() ) {
		throw std::invalid_argument(
		    "Tensor::assignTensor(): size must be equal to the number of elements in the tensor" );
	}
	
	std::copy( value, value + size, m_data );
}

template <typename T>
void Tensor<T>::assign( value_type const *data, std::size_t size )
{
	if ( size != this->size() ) {
		throw std::invalid_argument(
		    "Tensor::assign(): size does not match tensor size" );
	}
	std::copy( data, data + size, m_data );
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
	map_ref() = map_ref().transpose().eval();
	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::reshape( std::size_t rows, std::size_t cols ) const
{
	return Tensor<T>( map_ref().template reshaped<Eigen::RowMajor>( rows, cols ) );
}

template <typename T>
Tensor<T> &Tensor<T>::reshapeInPlace( std::size_t rows,
                                      std::size_t cols )
{
	if ( rows * cols != size() ) {
		throw std::invalid_argument(
		    "Tensor::reshapeInPlace(): rows*cols must equal the current element count" );
	}
	map_ref() = map_ref().template reshaped<Eigen::RowMajor>( rows, cols ).eval();
	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::matmul( Tensor const &other ) const
{
	return Tensor<T>( ( map_ref() * other.map_ref() ).eval() );
}

template <typename T>
Tensor<T> &Tensor<T>::matmulInPlace( Tensor const &other )
{
	map_ref() = ( map_ref() * other.map_ref() ).eval();
	return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::matmulInPlace( Tensor const &m1, Tensor const &m2, bool transpose_first,
                                     bool transpose_second )
{
	auto const throw_dim = [] {
		throw std::invalid_argument(
		    "Tensor::matmulInPlace(m1,m2,transpose_first,transpose_second): incompatible dimensions" );
	};
	auto const throw_alias = [] {
		throw std::runtime_error(
		    "Tensor::matmulInPlace(m1,m2,transpose_first,transpose_second): self-aliasing is not supported" );
	};

	if ( transpose_first && transpose_second ) {
		if ( m1.rows() != m2.cols() || m1.cols() != rows() || m2.rows() != cols() ) {
			throw_dim();
		}
		if ( this != &m1 && this != &m2 ) {
			map_ref() = ( m1.map_ref().transpose() * m2.map_ref().transpose() ).eval();
		} else {
			throw_alias();
		}
		return *this;
	}
	if ( transpose_first ) {
		if ( m1.rows() != m2.rows() || m1.cols() != rows() || m2.cols() != cols() ) {
			throw_dim();
		}
		if ( this != &m1 && this != &m2 ) {
			map_ref() = ( m1.map_ref().transpose() * m2.map_ref() ).eval();
		} else {
			throw_alias();
		}
		return *this;
	}
	if ( transpose_second ) {
		if ( m1.cols() != m2.cols() || m1.rows() != rows() || m2.rows() != cols() ) {
			throw_dim();
		}
		if ( this != &m1 && this != &m2 ) {
			map_ref() = ( m1.map_ref() * m2.map_ref().transpose() ).eval();
		} else {
			throw_alias();
		}
		return *this;
	}
	if ( m1.cols() != m2.rows() || m1.rows() != rows() || m2.cols() != cols() ) {
		throw_dim();
	}
	if ( this != &m1 && this != &m2 ) {
		map_ref() = ( m1.map_ref() * m2.map_ref() ).eval();
	} else {
		throw_alias();
	}

	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::divideRowsWithCol( Tensor const &other ) const
{
	return Tensor<T>(
	    ( map_ref().array().colwise() / other.map_ref().col( 0 ).array() ).matrix().eval() );
}

template <typename T>
Tensor<T> &Tensor<T>::divideRowsWithColInPlace( Tensor const &other )
{
	map_ref().array().colwise() /= other.map_ref().col( 0 ).array();
	return *this;
}

template < typename T >
void Tensor<T>::elementwiseMultiply( Tensor const &other, Tensor &out ) const
{
	if ( rows() != other.rows() || cols() != other.cols() ) {
		throw std::invalid_argument(
		    "Tensor::elementwiseMultiply(): incompatible dimensions" );
	}
	if ( out.rows() != rows() || out.cols() != cols() ) {
		out.throw_if_non_owning_realloc(
		    "Tensor::elementwiseMultiply(): cannot resize a non-owning output tensor" );
		out = Tensor<T>( rows(), cols() );
	}
	matrix_type const product =
	    ( map_ref().array() * other.map_ref().array() ).matrix().eval();
	std::copy( product.data(), product.data() + product.size(), out.m_data );
}

template < typename T >
Tensor<T> &Tensor<T>::mulNSubstractInPlace( Tensor const &other, Tensor::value_type scalar )
{
	map_ref() -= other.map_ref()*scalar;
	
	return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::axpy( Tensor const &x, Tensor::value_type alpha )
{
	if ( rows() != x.rows() || cols() != x.cols() ) {
		throw std::invalid_argument( "Tensor::axpy(): incompatible dimensions" );
	}
	map_ref() += alpha * x.map_ref();
	return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::geam( value_type alpha, value_type beta, Tensor const &B )
{
	if ( rows() != B.rows() || cols() != B.cols() ) {
		throw std::invalid_argument( "Tensor::geam(): incompatible dimensions" );
	}
	map_ref() = alpha * map_ref() + beta * B.map_ref();
	return *this;
}

template <typename T>
typename Tensor<T>::value_type Tensor<T>::maxCoeff() const
{
	if ( map_ref().size() == 0 ) {
		return value_type{};
	}
	return map_ref().cwiseAbs().maxCoeff();
}

template <typename T>
Tensor<T> Tensor<T>::operator+( Tensor const &other ) const
{
	return Tensor<T>( ( map_ref() + other.map_ref() ).eval() );
}

template <typename T>
Tensor<T> &Tensor<T>::operator+=( Tensor const &other )
{
	map_ref() += other.map_ref();
	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::operator-( Tensor const &other ) const
{
	return Tensor<T>( ( map_ref() - other.map_ref() ).eval() );
}

template <typename T>
Tensor<T> &Tensor<T>::operator-=( Tensor const &other )
{
	map_ref() -= other.map_ref();
	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::addColwise( Tensor const &col ) const
{
	return Tensor<T>(
	    ( map_ref().rowwise() + col.map_ref().col( 0 ).transpose() ).eval() );
}

template <typename T>
Tensor<T> &Tensor<T>::addColwiseInPlace( Tensor const &col )
{
	map_ref().rowwise() += col.map_ref().col( 0 ).transpose();
	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::subtractColwise( Tensor const &col ) const
{
	return Tensor<T>(
	    ( map_ref().array().colwise() - col.map_ref().col( 0 ).array() ).matrix().eval() );
}

template <typename T>
Tensor<T> Tensor<T>::operator*( Tensor const &other ) const
{
	return Tensor<T>( ( map_ref().array() * other.map_ref().array() ).matrix().eval() );
}

template <typename T>
Tensor<T> &Tensor<T>::operator*=( Tensor const &other )
{
	map_ref().array() *= other.map_ref().array();
	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::operator*( value_type scalar ) const
{
	return Tensor<T>( ( map_ref() * scalar ).eval() );
}

template <typename T>
Tensor<T> &Tensor<T>::operator*=( value_type scalar )
{
	map_ref() *= scalar;
	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::cwiseGreater( value_type scalar ) const
{
	return Tensor<T>( ( map_ref().array() > scalar ).template cast<value_type>().matrix().eval() );
}

template <typename T>
Tensor<T> &Tensor<T>::cwiseGreaterInPlace( Tensor const &other, value_type scalar )
{
	if ( rows() != other.rows() || cols() != other.cols() ) {
		throw std::invalid_argument( "Tensor::cwiseGreaterInPlace(other, scalar): incompatible dimensions" );
	}
	map_ref() = ( other.map_ref().array() > scalar ).template cast<value_type>().matrix().eval();
	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::cwiseOneMinus() const
{
	return Tensor<T>( ( 1.0 - map_ref().array() ).matrix().eval() );
}

template <typename T>
void Tensor<T>::cwiseSigmoid( Tensor<T> &out ) const
{
	if ( out.rows() != rows() || out.cols() != cols() ) {
		out.throw_if_non_owning_realloc(
		    "Tensor::cwiseSigmoid(): cannot resize a non-owning output tensor" );
		out = Tensor<T>( rows(), cols() );
	}
	if ( rows() == 0 || cols() == 0 ) {
		return;
	}
	matrix_type const result =
	    ( 1.0 / ( 1.0 + ( -map_ref().array() ).exp() ) ).matrix().eval();
	std::copy( result.data(), result.data() + result.size(), out.m_data );
}

template <typename T>
Tensor<T> Tensor<T>::cwiseSigmoid() const
{
	Tensor<T> out;
	cwiseSigmoid( out );
	return out;
}

template <typename T>
Tensor<T> Tensor<T>::cwiseExp() const
{
	return Tensor<T>( ( map_ref().array().exp() ).matrix().eval() );
}

template <typename T>
Tensor<T> Tensor<T>::cwiseLog() const
{
	return Tensor<T>( ( map_ref().array().log() ).matrix().eval() );
}

template <typename T>
T Tensor<T>::sum() const
{
	if ( map_ref().size() == 0 ) {
		return value_type{};
	}
	return map_ref().sum();
}

template <typename T>
T Tensor<T>::asum() const
{
	if ( map_ref().size() == 0 ) {
		return value_type{};
	}
	return map_ref().cwiseAbs().sum();
}

template <typename T>
Tensor<T> Tensor<T>::sumAlongAxis( std::size_t axis, bool transpose_out ) const
{
	if ( axis != 0 && axis != 1 ) {
		throw std::invalid_argument( "Tensor::sumAlongAxis(): axis must be 0 or 1" );
	}
	if ( map_ref().size() == 0 ) {
		return Tensor<T>{};
	}
	if ( axis == 0 ) {
		if ( !transpose_out ) {
			return Tensor<T>( map_ref().colwise().sum() );
		}
		return Tensor<T>( map_ref().colwise().sum().transpose() );
	}
	if ( !transpose_out ) {
		return Tensor<T>( map_ref().rowwise().sum() );
	}
	return Tensor<T>( map_ref().rowwise().sum().transpose() );
}

template <typename T>
Tensor<T> &Tensor<T>::sumAlongAxisInPlace( Tensor const &src, std::size_t axis, bool transpose_out )
{
	if ( this == &src ) {
		throw std::runtime_error(
		    "Tensor::sumAlongAxisInPlace(src, axis): src must not be *this" );
	}
	if ( axis != 0 && axis != 1 ) {
		throw std::invalid_argument( "Tensor::sumAlongAxisInPlace(): axis must be 0 or 1" );
	}
	if ( src.map_ref().size() == 0 ) {
		if ( map_ref().size() == 0 ) {
			return *this;
		}
		throw std::invalid_argument( "Tensor::sumAlongAxisInPlace(): incompatible dimensions" );
	}
	if ( axis == 0 ) {
		if ( !transpose_out ) {
			if ( rows() != 1u || cols() != src.cols() ) {
				throw std::invalid_argument(
				    "Tensor::sumAlongAxisInPlace(): for axis 0, *this must have shape (1, src.cols())" );
			}
			map_ref() = src.map_ref().colwise().sum();
		} else {
			if ( rows() != src.cols() || cols() != 1u ) {
				throw std::invalid_argument(
				    "Tensor::sumAlongAxisInPlace(): for axis 0 with transpose_out, *this must have shape (src.cols(), 1)" );
			}
			map_ref() = src.map_ref().colwise().sum().transpose();
		}
		return *this;
	}
	if ( !transpose_out ) {
		if ( rows() != src.rows() || cols() != 1u ) {
			throw std::invalid_argument(
			    "Tensor::sumAlongAxisInPlace(): for axis 1, *this must have shape (src.rows(), 1)" );
		}
		map_ref() = src.map_ref().rowwise().sum();
	} else {
		if ( rows() != 1u || cols() != src.rows() ) {
			throw std::invalid_argument(
			    "Tensor::sumAlongAxisInPlace(): for axis 1 with transpose_out, *this must have shape (1, src.rows())" );
		}
		map_ref() = src.map_ref().rowwise().sum().transpose();
	}
	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::max_along_axis( std::size_t axis ) const
{
	if ( axis == 0 ) {
		return Tensor<T>( map_ref().colwise().maxCoeff() );
	}
	return Tensor<T>( map_ref().rowwise().maxCoeff() );
}

template <typename T>
Tensor<Eigen::Index> Tensor<T>::argmaxAlongAxis( std::size_t axis ) const
{
	if ( axis == 0 ) {
		Tensor<Eigen::Index> out( 1, cols() );
		for ( std::size_t j = 0; j < cols(); ++j ) {
			Eigen::Index idx = 0;
			map_ref().col( static_cast<Eigen::Index>( j ) ).maxCoeff( &idx );
			out.assign( 0, j, idx );
		}
		return out;
	}
	if ( axis == 1 ) {
		Tensor<Eigen::Index> out( rows(), 1 );
		for ( std::size_t i = 0; i < rows(); ++i ) {
			Eigen::Index idx = 0;
			map_ref().row( static_cast<Eigen::Index>( i ) ).maxCoeff( &idx );
			out.assign( i, 0, idx );
		}
		return out;
	}
	throw std::invalid_argument( "Tensor::argmaxAlongAxis(): axis must be 0 or 1" );
}

template <typename T>
template <typename Generator>
void Tensor<T>::randomize( Generator &generator ) noexcept
{
	std::generate( map_ref().data(), map_ref().data() + map_ref().size(), generator );
}

template <typename T>
void Tensor<T>::randomize( std::uint64_t seed ) noexcept
{
	std::mt19937_64 gen( seed );
	std::uniform_real_distribution<double> dis( 0.0, 1.0 );
	auto generator = [&]() { return static_cast<T>( dis( gen ) ); };
	randomize( generator );
}

template <typename T>
void Tensor<T>::randomizeHe( std::size_t fan_in, std::uint64_t seed ) noexcept
{
	double const scale = std::sqrt( 2.0 / static_cast<double>( fan_in ) );
	std::mt19937_64 gen( seed );
	std::uniform_real_distribution<double> dis( -scale, scale );
	auto generator = [&]() { return static_cast<T>( dis( gen ) ); };
	randomize( generator );
}

template <typename T>
void Tensor<T>::randomizePytorchDefault( std::size_t fan_in, std::uint64_t seed ) noexcept
{
	if ( fan_in == 0 ) {
		return;
	}
	double const inv_sqrt = 1.0 / std::sqrt( static_cast<double>( fan_in ) );
	std::mt19937_64 gen( seed );
	std::uniform_real_distribution<double> dis( -inv_sqrt, inv_sqrt );
	auto generator = [&]() { return static_cast<T>( dis( gen ) ); };
	randomize( generator );
}

/// Callable batch policy: holds flattened input width (\c input_cols), target row width, and batch row count
/// (use in custom \c operator() e.g. for augmentation). Default is a no-op after row gather.
template <typename T = float>
struct DefaultAssignBatchOp {
	std::size_t input_cols = 0;
	std::size_t target_cols = 0;
	std::size_t batch_rows = 0;

	constexpr DefaultAssignBatchOp() = default;
	constexpr DefaultAssignBatchOp( std::size_t in_cols, std::size_t out_cols, std::size_t rows )
	    : input_cols( in_cols ), target_cols( out_cols ), batch_rows( rows )
	{
	}

	void operator()( Tensor<T> &host_x, Tensor<T> &host_y ) const
	{
		(void)host_x;
		(void)host_y;
	}
};

template <typename T, typename Op>
inline void assignBatch(
    Tensor<T> &input_batch,
    Tensor<T> &target_batch,
    std::vector<Tensor<T>> const &host_inputs,
    std::vector<Tensor<T>> const &host_targets,
    std::vector<int> const &batch_indices_int,
    std::size_t batch_window_start,
    std::size_t batch_size,
    Tensor<T> &host_x,
    Tensor<T> &host_y,
    Op const &op )
{
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for ( std::size_t r = 0; r < batch_size; ++r ) {
		int const idx = batch_indices_int[static_cast<std::size_t>( batch_window_start + r )];
		host_x.assignTensorAsRow( r, host_inputs[static_cast<std::size_t>( idx )] );
		host_y.assignTensorAsRow( r, host_targets[static_cast<std::size_t>( idx )] );
	}

	op( host_x, host_y );

	if ( std::addressof( input_batch ) != std::addressof( host_x ) ) {
		input_batch.assign( host_x.data(), host_x.size() );
	}
	if ( std::addressof( target_batch ) != std::addressof( host_y ) ) {
		target_batch.assign( host_y.data(), host_y.size() );
	}
}

}

#endif