#ifndef LAYER_PROPERTIES_HPP
#define LAYER_PROPERTIES_HPP
#include "tensor_like.hpp"
#include <concepts>

template <typename Layer, typename Tensor>
concept LayerLike = TensorLike<Tensor> && requires( Layer l ) {
	{ l.forward() } -> std::convertible_to<Tensor *>;
	{ l.backward() } -> std::convertible_to<Tensor *>;
};

template < typename Layer, typename Tensor >
concept UpdateableLayer = LayerLike<Layer, Tensor> && requires( Layer l, Tensor &param, Tensor &grad ) {
	{ l.getGradWeights() } -> std::convertible_to<Tensor>;
	{ l.getGradBias() } -> std::convertible_to<Tensor>;
};

#endif
