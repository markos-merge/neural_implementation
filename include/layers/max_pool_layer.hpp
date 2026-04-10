#ifndef MAX_POOL_LAYER_HPP
#define MAX_POOL_LAYER_HPP

#include <cstddef>
#include <array>
#include <limits>
#include "layer_base.hpp"

namespace neural {

template <typename TensorN_t>
class MaxPoolLayer : public LayerBase<TensorN_t>
{
	public:
		using tensor_t = TensorN_t;
	public:
		MaxPoolLayer( std::size_t pool_size, std::size_t stride );

		std::size_t poolSize() const { return m_pool_size; }
		std::size_t stride() const { return m_stride; }

		TensorN_t *forward();
		TensorN_t *backward();

	private:
		using argmax_tensor_type = typename TensorN_t::template tensor_alias<
		    4, std::size_t>;

		std::size_t m_pool_size;
		std::size_t m_stride;

		argmax_tensor_type m_argmax;
};

template <typename TensorN_t>
MaxPoolLayer<TensorN_t>::MaxPoolLayer( std::size_t pool_size, std::size_t stride )
	: m_pool_size( pool_size )
	, m_stride( stride )
{
}

template <typename TensorN_t>
TensorN_t *MaxPoolLayer<TensorN_t>::forward()
{
	std::array< std::size_t, 4 > output_shape = this->getInput()->shape();
	output_shape[3] = (this->getInput()->shape()[3] - m_pool_size ) / m_stride + 1;
	output_shape[2] = (this->getInput()->shape()[2] - m_pool_size ) / m_stride + 1;
	if( m_argmax.shape() != output_shape ) {
		m_argmax = argmax_tensor_type( output_shape );
	}
	
	if( this->getOutput()->shape() != output_shape ) {
		*this->getOutput() = TensorN_t( output_shape );
	}

	#ifdef _OPENMP
	#pragma omp parallel for schedule( static )
	#endif
	for ( std::size_t b = 0; b < this->getInput()->shape()[0]; ++b ) {
		for ( std::size_t c = 0; c < this->getInput()->shape()[1]; ++c ) {
			for ( std::size_t oh = 0; oh < output_shape[2]; ++oh ) {
				for ( std::size_t ow = 0; ow < output_shape[3]; ++ow ) {
					std::size_t const h_start = oh * m_stride;
					std::size_t const w_start = ow * m_stride;
					typename TensorN_t::value_type max = std::numeric_limits<typename TensorN_t::value_type>::lowest();
					for ( std::size_t k = 0; k < m_pool_size; ++k ) {
						for ( std::size_t l = 0; l < m_pool_size; ++l ) {
							if ( this->getInput()->at( b, c, h_start + k, w_start + l ) > max ) {
								max = this->getInput()->at( b, c, h_start + k, w_start + l );
								m_argmax.at( b, c, oh, ow ) = k * m_pool_size + l;
							}
						}
					}
					this->getOutput()->at( b, c, oh, ow ) = max;
				}
			}
		}
	}

	return this->getOutput();
}

template <typename TensorN_t>
TensorN_t *MaxPoolLayer<TensorN_t>::backward()
{
	if( this->getGradOutput()->shape() != this->getInput()->shape() ) {
		*this->getGradOutput() = TensorN_t( this->getInput()->shape() );
	}
	std::array< std::size_t, 4 > grad_input_shape = this->getOutput()->shape();
	
	typename TensorN_t::value_type *output_data = this->getGradOutput()->data();
	std::fill( output_data, output_data + this->getGradOutput()->size(), 0.0 );

	#ifdef _OPENMP
	#pragma omp parallel for schedule( static )
	#endif
	for( std::size_t b = 0; b < grad_input_shape[0]; ++b ) {
		for( std::size_t c = 0; c < grad_input_shape[1]; ++c ) {
			for( std::size_t oh = 0; oh < grad_input_shape[2]; ++oh ) {
				for( std::size_t ow = 0; ow < grad_input_shape[3]; ++ow ) {
					std::size_t const h_start = oh * m_stride;
					std::size_t const w_start = ow * m_stride;
					std::size_t const k = m_argmax.at( b, c, oh, ow ) / m_pool_size;
					std::size_t const l = m_argmax.at( b, c, oh, ow ) % m_pool_size;
					typename TensorN_t::value_type val = this->getGradInput()->at( b, c, oh, ow );
					this->getGradOutput()->addElementwise( { b, c, h_start + k, w_start + l }, val );
				}
			}
		}
	}

	return this->getGradOutput();
}

} // namespace neural

#endif
