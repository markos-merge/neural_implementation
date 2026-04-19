#ifndef MOMENTUM_SGD_OPTIMIZER_HPP
#define MOMENTUM_SGD_OPTIMIZER_HPP

#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <functional>
#include <type_traits>
#include <utility>
#include <future>
#include "convolutional_box.hpp"
#include "layer.hpp"
#include "tensor.hpp"
#include <tuple>

#if NEURAL_CUDA_ENABLED
#include "cuda_tensor.hpp"
#endif

namespace neural {

/// Flattens a layer-type pack: if a type has nested \c tensor_tuple_t (e.g. ConvolutionalBox),
/// its inner \c std::tuple<Inner...> is expanded; otherwise the type is kept as \c std::tuple<type>.
/// Results are concatenated left-to-right (same order as \c TupleTypeFinder<type, types...>).

template <typename... types>
struct TupleTypeFinder;

template <>
struct TupleTypeFinder<> {
	using tuple_type = std::tuple<>;
};

template <typename Tuple>
struct TupleTypeFinderFromTuple;

template <typename... Us>
struct TupleTypeFinderFromTuple<std::tuple<Us...>> {
	using tuple_type = typename TupleTypeFinder<Us...>::tuple_type;
};

/// Convenience when you only have \c std::tuple<Layers...> (e.g. \c NN::tensor_tuple_t).
template <typename Tuple>
using TupleTypeFinderFromTuple_t = typename TupleTypeFinderFromTuple<Tuple>::tuple_type;

template <typename, typename = void>
struct has_tensor_tuple_impl : std::false_type {};

template <typename T>
struct has_tensor_tuple_impl<T, std::void_t<typename T::tensor_tuple_t>> : std::true_type {};

template <typename T>
inline constexpr bool has_tensor_tuple_v = has_tensor_tuple_impl<T>::value;

template <typename type, typename tuple_t, bool HasNested = has_tensor_tuple_v<type>>
struct TupleTypeAdderImpl;

/// Expand \c type::tensor_tuple_t (e.g. inside ConvolutionalBox).
template <typename type, typename tuple_t>
struct TupleTypeAdderImpl<type, tuple_t, true> {
	using tuple_type = decltype( std::tuple_cat(
	    std::declval<typename TupleTypeFinderFromTuple<typename type::tensor_tuple_t>::tuple_type>(),
	    std::declval<tuple_t>() ) );
};

/// Single layer type (no nested \c tensor_tuple_t).
template <typename type, typename tuple_t>
struct TupleTypeAdderImpl<type, tuple_t, false> {
	using tuple_type = decltype(
	    std::tuple_cat( std::declval<std::tuple<type>>(), std::declval<tuple_t>() ) );
};

template <typename type, typename tuple_t>
struct TupleTypeAdder : TupleTypeAdderImpl<type, tuple_t> {};

template <typename type, typename... types>
struct TupleTypeFinder<type, types...> {
	using tuple_type =
	    typename TupleTypeAdder<type, typename TupleTypeFinder<types...>::tuple_type>::tuple_type;
};

template <typename Layer>
concept HasGetWeights = requires( Layer &l ) {
	l.getWeights();
};

template <typename Layer>
concept HasGetBias = requires( Layer &l ) {
	l.getBias();
};

/// Placeholder when a layer has no weight/bias velocity slots.
struct EmptyMomentumSlot {};

template <typename Layer>
struct VelocityChunk;

namespace detail {

template <typename Tuple>
struct InnerVelocityFromTuple;

template <typename... Inner>
struct InnerVelocityFromTuple<std::tuple<Inner...>> {
	using type = std::tuple<VelocityChunk<Inner>...>;
};

template <typename Layer, bool HasNested = has_tensor_tuple_v<Layer>>
struct VelocityChunkInnerHelper {
	using inner_velocity_tuple_t = std::tuple<>;
};

template <typename Layer>
struct VelocityChunkInnerHelper<Layer, true> {
	using inner_velocity_tuple_t =
	    typename InnerVelocityFromTuple<typename Layer::tensor_tuple_t>::type;
};

template <typename Layer, bool = HasGetWeights<Layer>>
struct VelocityChunkWeightsHelper {
	using type = EmptyMomentumSlot;
};

template <typename Layer>
struct VelocityChunkWeightsHelper<Layer, true> {
	using type = std::decay_t<decltype( std::declval<Layer &>().getWeights() )>;
};

template <typename Layer, bool = HasGetBias<Layer>>
struct VelocityChunkBiasHelper {
	using type = EmptyMomentumSlot;
};

template <typename Layer>
struct VelocityChunkBiasHelper<Layer, true> {
	using type = std::decay_t<decltype( std::declval<Layer &>().getBias() )>;
};

} // namespace detail

template <typename Layer>
struct VelocityChunk {
	using inner_velocity_tuple_t = typename detail::VelocityChunkInnerHelper<Layer>::inner_velocity_tuple_t;
	inner_velocity_tuple_t m_inner{};

	using weights_tensor_t = typename detail::VelocityChunkWeightsHelper<Layer>::type;
	using bias_tensor_t    = typename detail::VelocityChunkBiasHelper<Layer>::type;

	weights_tensor_t m_vel_weights{};
	bias_tensor_t m_vel_bias{};
};

template <typename Tuple>
struct VelocityTuple;

template <typename... Layers>
struct VelocityTuple<std::tuple<Layers...>> {
	using tuple_type = std::tuple<VelocityChunk<Layers>...>;
};

template <typename Tensor, typename NN>
class MomentumSGDOptimizer
{
	public:
		using tuple_type_t       = typename NN::tensor_tuple_t;
		using velocity_storage_t = typename VelocityTuple<tuple_type_t>::tuple_type;
	public:
		/// Return true to request early stop; false continues training.
		using ProgressCallback =
		    std::function<bool( std::size_t epoch, std::size_t epoch_max,
		                        std::size_t batch_idx, std::size_t batch_max,
		                        typename Tensor::value_type loss )>;

		MomentumSGDOptimizer( NN &nn );

		typename Tensor::value_type m_learning_rate = static_cast<typename Tensor::value_type>(0.01);
		/// Momentum coefficient \(\beta\) for \(v_t = \beta v_{t-1} + \nabla L\).
		typename Tensor::value_type m_momentum =
		    static_cast<typename Tensor::value_type>(0.9);
		/// \(\lambda\) for \(\frac{\lambda}{2}\|W\|^2\) on **weights only** (velocity gets \(+\lambda W\); biases unchanged). Use \c 0 to disable. If training stalls or diverges, lower \(\lambda\) or \c m_learning_rate.
		typename Tensor::value_type m_l2_regularizer = static_cast<typename Tensor::value_type>( 0 );
		std::size_t m_batch_size = 50;
		std::size_t m_epochs = 100;

		/// Dataset rows are **host** \c ::neural::Tensor<float> (one row per sample). For \c CudaTensor<float>
		/// networks, each batch is gathered on the CPU then copied to device latch buffers.
		void train( std::vector<::neural::Tensor<float>> &inputs,
		            std::vector<::neural::Tensor<float>> &targets,
		            ProgressCallback callback = nullptr );

		template <typename Op>
		void train( std::vector<::neural::Tensor<float>> &inputs,
		            std::vector<::neural::Tensor<float>> &targets,
		            ProgressCallback callback,
		            Op const &op );
	private:
		void initialize();
		void applyStep();
		void initMomentumState();
	private:
		NN &m_nn;
		velocity_storage_t m_velocity;
};

namespace detail {

/// Momentum with optional L2: \(v \leftarrow \beta v + g + \lambda\theta\), then \(\theta \leftarrow \theta - \eta v\)
/// (equivalent to task gradient \(g\) plus \(\lambda\theta\) from \(\frac{\lambda}{2}\|\theta\|^2\)).
template <typename ParamTensor, typename Layer>
void momentum_updateParam_impl( VelocityChunk<Layer> &chunk, Layer &layer,
                                typename ParamTensor::value_type momentum,
                                typename ParamTensor::value_type learning_rate,
                                typename ParamTensor::value_type l2_regularizer )
{
	if constexpr ( HasGetWeights<Layer> ) {
		auto &v = chunk.m_vel_weights;
		auto &w = layer.getWeights();
		auto const &g = layer.getGradWeights();
		v *= momentum;
		// \(v \mathrel{+}= g\) via \(v - (-1)\cdot g\)
		v.mulNSubstractInPlace( g, static_cast<typename std::decay_t<decltype( v )>::value_type>( -1 ) );
		if ( l2_regularizer > static_cast<typename ParamTensor::value_type>( 0 ) ) {
			// \(v \mathrel{+}= \lambda w\) via \(v - (-\lambda) w\)
			v.mulNSubstractInPlace( w, -l2_regularizer );
		}
		w.mulNSubstractInPlace( v, learning_rate );
	}

	if constexpr ( HasGetBias<Layer> ) {
		auto &vb = chunk.m_vel_bias;
		auto &b  = layer.getBias();
		auto const &gb = layer.getGradBias();
		vb *= momentum;
		vb.mulNSubstractInPlace( gb, static_cast<typename std::decay_t<decltype( vb )>::value_type>( -1 ) );
		b.mulNSubstractInPlace( vb, learning_rate );
	}
}

/// \p ParamTensor matches \c Layer's parameter tensors (e.g. \c Tensor2D for linear heads,
/// \c TensorN for convolutional layers inside a box), same convention as \c SGDOptimizer.
template <typename ParamTensor, typename Layer>
void momentum_updateLayer_impl( Layer &layer, VelocityChunk<Layer> &chunk,
                                typename ParamTensor::value_type learning_rate,
                                typename ParamTensor::value_type momentum,
                                typename ParamTensor::value_type l2_regularizer )
{
	if constexpr ( has_tensor_tuple_v<std::decay_t<Layer>> ) {
		layer.forEachLayerZip( chunk.m_inner, [&]( auto &sub, auto &sub_chunk ) {
			using Sub = std::decay_t<decltype( sub )>;
			momentum_updateLayer_impl<typename Sub::tensor_t, Sub>(
			    sub, sub_chunk, learning_rate, momentum, l2_regularizer );
		} );
	} else if constexpr ( UpdateableLayer<std::decay_t<Layer>, ParamTensor> ) {
		momentum_updateParam_impl<ParamTensor, Layer>( chunk, layer, momentum, learning_rate,
		                                               l2_regularizer );
	}
}

template <typename Layer, typename Chunk>
void init_velocity_chunk( Layer &layer, Chunk &chunk )
{
	if constexpr ( has_tensor_tuple_v<std::decay_t<Layer>> ) {
		layer.forEachLayerZip( chunk.m_inner, []( auto &sub, auto &sub_chunk ) {
			init_velocity_chunk( sub, sub_chunk );
		} );
	}
	if constexpr ( HasGetWeights<std::decay_t<Layer>> ) {
		auto const &w = layer.getWeights();
		using W       = std::decay_t<decltype( w )>;
		if constexpr ( requires { w.rows(); w.cols(); } ) {
			chunk.m_vel_weights = W( w.rows(), w.cols(), typename W::value_type{} );
		} else {
			chunk.m_vel_weights = W( w.shape(), typename W::value_type{} );
		}
	}
	if constexpr ( HasGetBias<std::decay_t<Layer>> ) {
		auto const &b = layer.getBias();
		using B       = std::decay_t<decltype( b )>;
		if constexpr ( requires { b.rows(); b.cols(); } ) {
			chunk.m_vel_bias = B( b.rows(), b.cols(), typename B::value_type{} );
		} else {
			chunk.m_vel_bias = B( b.shape(), typename B::value_type{} );
		}
	}
}
} // namespace detail

template <typename Tensor, typename NN>
MomentumSGDOptimizer<Tensor, NN>::MomentumSGDOptimizer( NN &nn )
	: m_nn( nn )
{
}

template <typename Tensor, typename NN>
void MomentumSGDOptimizer<Tensor, NN>::initMomentumState()
{
	m_nn.forEachLayerZip( m_velocity, []( auto &layer, auto &chunk ) {
		detail::init_velocity_chunk( layer, chunk );
	} );
}

template <typename Tensor, typename NN>
void MomentumSGDOptimizer<Tensor, NN>::applyStep()
{
	m_nn.forEachLayerZip( m_velocity, [this]( auto &layer, auto &chunk ) {
		detail::momentum_updateLayer_impl<Tensor>( layer, chunk, this->m_learning_rate,
		                                           this->m_momentum, this->m_l2_regularizer );
	} );
}

template< typename Tensor, typename NN >
void MomentumSGDOptimizer<Tensor, NN>::initialize()
{
	m_nn.forEachLayer( [this]( auto &&layer ) {
		if constexpr ( requires { layer.initialize(); } ) {
			layer.initialize();
		}
	} );
	initMomentumState();
}

template< typename Tensor_t, typename NN >
void MomentumSGDOptimizer<Tensor_t, NN>::train( std::vector<::neural::Tensor<float>> &inputs,
                                                std::vector<::neural::Tensor<float>> &targets,
                                                typename MomentumSGDOptimizer<Tensor_t, NN>::ProgressCallback callback )
{
	std::size_t const total_batches = inputs.size() / m_batch_size;
	if ( total_batches == 0 ) {
		return;
	}
	std::size_t const in_cols  = inputs[0].cols();
	std::size_t const out_cols = targets[0].cols();
	DefaultAssignBatchOp<float> const op( in_cols, out_cols, m_batch_size );
	train( inputs, targets, callback, op );
}

template< typename Tensor_t, typename NN >
template <typename Op>
void MomentumSGDOptimizer<Tensor_t, NN>::train( std::vector<::neural::Tensor<float>> &inputs,
                                                std::vector<::neural::Tensor<float>> &targets,
                                                typename MomentumSGDOptimizer<Tensor_t, NN>::ProgressCallback callback,
                                                Op const &op )
{
	std::size_t const total_batches = inputs.size() / m_batch_size;
	if ( total_batches == 0 ) {
		return;
	}
	std::vector< int > batch_indices_int( inputs.size() );
	std::size_t const in_cols  = inputs[0].cols();
	std::size_t const out_cols = targets[0].cols();

	m_nn.ensureBuffersForShape( m_batch_size, in_cols );

	::neural::Tensor<float> host_x0( m_batch_size, in_cols );
	::neural::Tensor<float> host_y0( m_batch_size, out_cols );
	::neural::Tensor<float> host_x1( m_batch_size, in_cols );
	::neural::Tensor<float> host_y1( m_batch_size, out_cols );
	
	bool momentum_initialized = false;

	for ( std::size_t epoch = 0; epoch < m_epochs; ++epoch ) {
		bool stop_training = false;
		std::vector<std::size_t> batch_indices( inputs.size() );
		std::iota( batch_indices.begin(), batch_indices.end(), 0 );
		std::shuffle( batch_indices.begin(), batch_indices.end(),
		              std::mt19937( static_cast<std::mt19937::result_type>( epoch ) ) );
		for ( std::size_t r = 0; r < batch_indices.size(); ++r ) {
			batch_indices_int[static_cast<std::size_t>( r )] =
			    static_cast<int>( batch_indices[static_cast<std::size_t>( r )] );
		}

		assignBatch( host_x0, host_y0, inputs, targets, batch_indices_int, 0, m_batch_size, host_x0,
		             host_y0, op );
		
		std::future<void> prefetch;
		std::size_t batch_idx = 0;
		typename Tensor_t::value_type loss = 0.0;

		for ( std::size_t k = 0; k < total_batches; ++k ) {
			if ( k > 0 ) {
				prefetch.wait();
			}
			int const cur = static_cast<int>( k % 2 );
			::neural::Tensor<float> &hx = cur == 0 ? host_x0 : host_x1;
			::neural::Tensor<float> &hy = cur == 0 ? host_y0 : host_y1;

			m_nn.inputLatch().input()->assign( hx.data(), hx.size() );
			m_nn.targetBuffer().assign( hy.data(), hy.size() );

			if ( k + 1 < total_batches ) {
				std::size_t const ni = ( k + 1 ) * m_batch_size;
				int const nxt = 1 - cur;
				prefetch = std::async( std::launch::async, [&, nxt, ni]() {
					if ( nxt == 0 ) {
						assignBatch( host_x0, host_y0, inputs, targets, batch_indices_int, ni,
						             m_batch_size, host_x0, host_y0, op );
					} else {
						assignBatch( host_x1, host_y1, inputs, targets, batch_indices_int, ni,
						             m_batch_size, host_x1, host_y1, op );
					}
				} );
			}

			loss = m_nn.trainStep();

			if ( !momentum_initialized ) {
				initMomentumState();
				momentum_initialized = true;
			}

			applyStep();

			if ( callback && callback( epoch, m_epochs, batch_idx, total_batches, loss ) ) {
				if ( prefetch.valid() ) {
					prefetch.wait();
				}
				stop_training = true;
				break;
			}
			++batch_idx;
		}
		if ( prefetch.valid() ) {
			prefetch.wait();
		}
		if ( stop_training ) {
			break;
		}
	}

	m_nn.ensureBuffersForShape( m_batch_size, in_cols );
}


} // namespace neural

#endif
