#ifndef REFACTOR_SEQUENTIAL_NN_HPP
#define REFACTOR_SEQUENTIAL_NN_HPP

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>
#include "device.hpp"
#include "latch_layer.hpp"
#include "layer.hpp"
#if NEURAL_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace neural {
namespace refactor {

// Runtime-polymorphic sequential network.
// Latches are wired after all layers are added via wire().
// forward() copies host data into the first forward slot, then runs each layer.
// backward() copies host gradient into the last gradient slot, then runs layers in reverse.
template <typename T, typename Device = Cpu>
class SequentialNN2
{
public:
	void addLayer( Layer<T, Device> layer );

	// Must be called once after all addLayer() calls, before forward/backward.
	void wire();

	// data: host pointer, N samples, in_features per sample (flat or first dim = N).
	void forward( T const *data, std::size_t N, std::size_t in_features );

	// grad_output: host pointer — pre-computed dL/d(last output), same size as output().
	void backward( T const *grad_output );

	// Raw pointer into the last forward slot (device memory when Device == Cuda).
	T *output();

	// Total number of elements in the output slot.
	std::size_t outputSize() const;

	// Shape of the output slot.
	std::vector<std::size_t> outputShape() const;

	template <typename F>
	void forEachLayer( F &&f );

private:
	std::vector<Layer<T, Device>>      m_layers;
	std::vector<LatchLayer<T, Device>> m_fwd_slots;   // size = m_layers.size() + 1
	std::vector<LatchLayer<T, Device>> m_grad_slots;  // size = m_layers.size() + 1
};

// ---------------------------------------------------------------------------

template <typename T, typename Device>
void SequentialNN2<T, Device>::addLayer( Layer<T, Device> layer )
{
	m_layers.push_back( std::move( layer ) );
}

template <typename T, typename Device>
void SequentialNN2<T, Device>::wire()
{
	std::size_t const n = m_layers.size();
	m_fwd_slots.resize( n + 1 );
	m_grad_slots.resize( n + 1 );

	for ( std::size_t i = 0; i < n; ++i ) {
		std::visit(
		    [&]( auto &l ) {
			    l.setInputOutputLatches( &m_fwd_slots[i], &m_fwd_slots[i + 1] );
			    l.setGradLatches( &m_grad_slots[i + 1], &m_grad_slots[i] );
		    },
		    m_layers[i] );
	}
}

template <typename T, typename Device>
void SequentialNN2<T, Device>::forward( T const *data, std::size_t N, std::size_t in_features )
{
	m_fwd_slots[0].resize( { N, in_features } );

	if constexpr ( std::is_same_v<Device, Cpu> ) {
		std::memcpy( m_fwd_slots[0].fwdData(), data, N * in_features * sizeof( T ) );
	}
#if NEURAL_CUDA_ENABLED
	else {
		cudaMemcpy( m_fwd_slots[0].fwdData(), data, N * in_features * sizeof( T ),
		            cudaMemcpyHostToDevice );
	}
#endif

	for ( auto &l : m_layers )
		std::visit( []( auto &layer ) { layer.forward(); }, l );
}

template <typename T, typename Device>
void SequentialNN2<T, Device>::backward( T const *grad_output )
{
	auto const &out_shape = m_fwd_slots.back().shape();
	m_grad_slots.back().resize( out_shape );

	std::size_t const total = std::accumulate( out_shape.begin(), out_shape.end(),
	                                            std::size_t{ 1 }, std::multiplies<std::size_t>() );

	if constexpr ( std::is_same_v<Device, Cpu> ) {
		std::memcpy( m_grad_slots.back().fwdData(), grad_output, total * sizeof( T ) );
	}
#if NEURAL_CUDA_ENABLED
	else {
		cudaMemcpy( m_grad_slots.back().fwdData(), grad_output, total * sizeof( T ),
		            cudaMemcpyHostToDevice );
	}
#endif

	for ( std::size_t i = m_layers.size(); i > 0; --i )
		std::visit( []( auto &layer ) { layer.backward(); }, m_layers[i - 1] );
}

template <typename T, typename Device>
T *SequentialNN2<T, Device>::output()
{
	return m_fwd_slots.back().fwdData();
}

template <typename T, typename Device>
std::size_t SequentialNN2<T, Device>::outputSize() const
{
	auto const &s = m_fwd_slots.back().shape();
	return std::accumulate( s.begin(), s.end(), std::size_t{ 1 }, std::multiplies<std::size_t>() );
}

template <typename T, typename Device>
std::vector<std::size_t> SequentialNN2<T, Device>::outputShape() const
{
	return m_fwd_slots.back().shape();
}

template <typename T, typename Device>
template <typename F>
void SequentialNN2<T, Device>::forEachLayer( F &&f )
{
	for ( auto &l : m_layers )
		std::visit( std::forward<F>( f ), l );
}

} // namespace refactor
} // namespace neural

#endif // REFACTOR_SEQUENTIAL_NN_HPP
