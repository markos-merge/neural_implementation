#ifndef NEURAL_IMPL_CUBLAS_HANDLE_HPP
#define NEURAL_IMPL_CUBLAS_HANDLE_HPP

#include <cublas_v2.h>

namespace neural {

class CublasHandle
{
	public:
		CublasHandle();
		~CublasHandle();

		static cublasHandle_t &instance();

	private:
		static cublasHandle_t m_cublas_handle;
};

} // namespace neural

#endif // NEURAL_IMPL_CUBLAS_HANDLE_HPP
