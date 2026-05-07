#ifndef REFACTOR_LATCH_LAYER_HPP
#define REFACTOR_LATCH_LAYER_HPP

#include <cstddef>
#include <vector>
#include "latch_base.hpp"
#include <numeric>
#include <stdexcept>
#if NEURAL_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace neural {
namespace refactor {

// CPU specialization (primary template)
template <typename T, typename Device = Cpu>
class LatchLayer : public LatchBase<T, Device>
{
public:
	T*                                fwdData() override;
	T*                                bwdData() override;
	std::vector<std::size_t> const&   shape()    const override;
	void resize(std::vector<std::size_t> const& s) override;

private:
	std::vector<std::size_t> m_shape;
	std::vector<T>           m_fwd;
	std::vector<T>           m_bwd;
};

// CUDA specialization
template <typename T>
class LatchLayer<T, Cuda> : public LatchBase<T, Cuda>
{
public:
	LatchLayer();
	~LatchLayer() override;

	T*                                fwdData() override;
	T*                                bwdData() override;
	std::vector<std::size_t> const&   shape()    const override;
	void resize(std::vector<std::size_t> const& s) override;

private:
	std::vector<std::size_t> m_shape;
	T* m_fwd = nullptr;
	T* m_bwd = nullptr;
};

//Implementation
//Cpu specialization
template <typename T, typename Device>
T* LatchLayer<T, Device>::fwdData()
{
	return m_fwd.data();
}

template <typename T, typename Device>
T* LatchLayer<T, Device>::bwdData()
{
	return m_bwd.data();
}

template <typename T, typename Device>
std::vector<std::size_t> const& LatchLayer<T, Device>::shape() const
{
	return m_shape;
}

template <typename T, typename Device>
void LatchLayer<T, Device>::resize(std::vector<std::size_t> const& s)
{
	m_shape = s;
	std::size_t const size = std::accumulate(s.begin(), s.end(), 1, std::multiplies<std::size_t>());
	m_fwd.resize(size);
	m_bwd.resize(size);
}

//Cuda specialization
template <typename T>
LatchLayer<T, Cuda>::LatchLayer()
{
}

template <typename T>
LatchLayer<T, Cuda>::~LatchLayer()
{
	if (m_fwd != nullptr ) {
		cudaFree(m_fwd);
	}

	if (m_bwd != nullptr ) {
		cudaFree(m_bwd);
	}

	m_fwd = nullptr;
	m_bwd = nullptr;
	m_shape.clear();
}

template <typename T>
T* LatchLayer<T, Cuda>::fwdData()
{
	return m_fwd;
}

template <typename T>
T* LatchLayer<T, Cuda>::bwdData()
{
	return m_bwd;
}

template <typename T>
std::vector<std::size_t> const& LatchLayer<T, Cuda>::shape() const
{
	return m_shape;
}

template <typename T>
void LatchLayer<T, Cuda>::resize(std::vector<std::size_t> const& s)
{
	std::size_t const size = std::accumulate(s.begin(), s.end(), 1, std::multiplies<std::size_t>());
	std::size_t const prv_size = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<std::size_t>());
	m_shape = s;
	if ( prv_size != size ) {
		if ( m_fwd != nullptr ) {
			cudaFree( m_fwd );
			m_fwd = nullptr;
		}
		if ( m_bwd != nullptr ) {
			cudaFree( m_bwd );
			m_bwd = nullptr;
		}
		cudaError_t cuda_stat = cudaMalloc( &m_fwd, size * sizeof( T ) );

		if ( cuda_stat != cudaSuccess ) {
			throw std::runtime_error( "cudaMalloc failed" );
		}

		cuda_stat = cudaMalloc( &m_bwd, size * sizeof( T ) );
		if ( cuda_stat != cudaSuccess ) {
			throw std::runtime_error( "cudaMalloc failed" );
		}
	}
}

} // namespace refactor
} // namespace neural

#endif // REFACTOR_LATCH_LAYER_HPP
