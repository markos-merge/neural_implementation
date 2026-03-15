#ifndef LAYER_HPP
#define LAYER_HPP
#include "tensor_like.hpp"

template <typename Layer, typename Tensor>
concept LayerLike = TensorLike<Tensor> && requires( Layer l, Tensor const &t ) {
	{ l.forward( t ) } -> std::convertible_to<Tensor>;
	{ l.backward( t ) } -> std::convertible_to<Tensor>;
};

#endif
