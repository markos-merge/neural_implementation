#ifndef NEURAL_IMPL_CUDNN_HANDLE_HPP
#define NEURAL_IMPL_CUDNN_HANDLE_HPP

#include <cudnn.h>

namespace neural {

class CudnnHandle
{
	public:
		CudnnHandle();
		~CudnnHandle();

		static cudnnHandle_t &instance();

	private:
		static cudnnHandle_t m_cudnn_handle;
};

} // namespace neural

#endif // NEURAL_IMPL_CUDNN_HANDLE_HPP
