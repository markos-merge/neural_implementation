#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#if NEURAL_CUDA_ENABLED
#include "cuda_tensor.hpp"
#include "tensor.hpp"
#endif

namespace neural {

template <typename Net, typename Tensor2D_t>
float compute_accuracy( Net &nn, std::vector<Tensor2D_t> const &images,
                        std::vector<Tensor2D_t> const &labels,
                        std::size_t num_classes = 10,
                        std::size_t batch_size = 1024 )
{
	if ( images.empty() )
		return 0.f;
	if ( batch_size == 0 )
		batch_size = images.size();

	std::size_t correct = 0;
	std::size_t const n = images.size();

	for ( std::size_t start = 0; start < n; start += batch_size ) {
		std::size_t const b = std::min( batch_size, n - start );
		Tensor2D_t inputs_tensor( b, images[0].size() );
		for ( std::size_t r = 0; r < b; ++r )
			inputs_tensor.assignTensorAsRow( r, images[start + r] );
		auto const out       = nn.forward( inputs_tensor );
		auto const pred_idxs = out.argmaxAlongAxis( 1 );
		for ( std::size_t i = 0; i < b; ++i ) {
			std::size_t const pred = static_cast<std::size_t>( pred_idxs( i, 0 ) );
			std::size_t gt = 0;
			for ( std::size_t c = 0; c < num_classes; ++c )
				if ( labels[start + i]( 0, c ) > 0.5f ) {
					gt = c;
					break;
				}
			if ( pred == gt )
				++correct;
		}
	}
	return static_cast<float>( correct ) / static_cast<float>( n );
}

#if NEURAL_CUDA_ENABLED
/// \p nn uses \c CudaTensor<float> activations; images and labels are **host** rows (e.g. from \ref load_cifar10_train).
template <typename Net>
float compute_accuracy( Net &nn, std::vector<Tensor<float>> const &host_images,
                        std::vector<Tensor<float>> const &host_labels,
                        std::size_t num_classes = 10,
                        std::size_t batch_size = 1024 )
{
	if ( host_images.empty() )
		return 0.f;
	if ( batch_size == 0 )
		batch_size = host_images.size();

	std::size_t correct = 0;
	std::size_t const n = host_images.size();
	std::size_t const cols = host_images[0].cols();

	for ( std::size_t start = 0; start < n; start += batch_size ) {
		std::size_t const b = std::min( batch_size, n - start );
		Tensor<float> host_batch( b, cols );
		for ( std::size_t r = 0; r < b; ++r )
			host_batch.assignTensorAsRow( r, host_images[start + r] );
		CudaTensor<float> inputs_tensor( b, cols );
		inputs_tensor.assign( host_batch.data(), host_batch.size() );
		auto const out       = nn.forward( inputs_tensor );
		auto const pred_idxs = out.argmaxAlongAxis( 1 );
		for ( std::size_t i = 0; i < b; ++i ) {
			std::size_t const pred = static_cast<std::size_t>( pred_idxs( i, 0 ) );
			std::size_t gt = 0;
			for ( std::size_t c = 0; c < num_classes; ++c )
				if ( host_labels[start + i]( 0, c ) > 0.5f ) {
					gt = c;
					break;
				}
			if ( pred == gt )
				++correct;
		}
	}
	return static_cast<float>( correct ) / static_cast<float>( n );
}
#endif

} // namespace neural
