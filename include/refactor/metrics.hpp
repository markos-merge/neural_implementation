#ifndef REFACTOR_METRICS_HPP
#define REFACTOR_METRICS_HPP

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>
#include "device.hpp"
#include "sequential_nn.hpp"
#include "tensor.hpp"
#if NEURAL_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace neural {
namespace refactor {

// Top-1 accuracy over a host dataset stored as per-sample Tensor<T> vectors.
// Runs inference in batches; handles both Cpu and Cuda devices.
template <typename T, typename Device>
float compute_accuracy( SequentialNN2<T, Device> &nn,
                        std::vector<neural::Tensor<T>> const &images,
                        std::vector<neural::Tensor<T>> const &labels,
                        std::size_t batch_size = 1024 )
{
	if ( images.empty() )
		return 0.f;

	std::size_t const n = images.size();
	std::size_t const in_cols = images[0].cols();
	std::size_t const n_cls = labels[0].cols();
	std::size_t correct = 0;

	std::vector<T> batch_x, host_out;

	for ( std::size_t start = 0; start < n; start += batch_size ) {
		std::size_t const b = std::min( batch_size, n - start );

		batch_x.resize( b * in_cols );
		for ( std::size_t i = 0; i < b; ++i )
			std::copy_n( images[start + i].data(), in_cols,
			             batch_x.data() + i * in_cols );

		nn.forward( batch_x.data(), b, in_cols );

		std::size_t const out_n = nn.outputSize();
		host_out.resize( out_n );

		if constexpr ( std::is_same_v<Device, Cpu> ) {
			std::memcpy( host_out.data(), nn.output(), out_n * sizeof( T ) );
		}
#if NEURAL_CUDA_ENABLED
		else {
			cudaMemcpy( host_out.data(), nn.output(), out_n * sizeof( T ),
			            cudaMemcpyDeviceToHost );
		}
#endif

		for ( std::size_t i = 0; i < b; ++i ) {
			T const *row = host_out.data() + i * n_cls;
			std::size_t pred = static_cast<std::size_t>(
			    std::max_element( row, row + n_cls ) - row );
			std::size_t gt = 0;
			for ( std::size_t c = 0; c < n_cls; ++c )
				if ( labels[start + i]( 0, c ) > T( 0.5 ) ) {
					gt = c;
					break;
				}
			if ( pred == gt )
				++correct;
		}
	}
	return static_cast<float>( correct ) / static_cast<float>( n );
}

} // namespace refactor
} // namespace neural

#endif // REFACTOR_METRICS_HPP
