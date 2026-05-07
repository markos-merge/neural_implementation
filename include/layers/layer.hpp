#ifndef LAYER_HPP
#define LAYER_HPP

#include <variant>
#include "batch_norm_1d_layer.hpp"
#include "convolutional_box.hpp"
#include "dropout_layer.hpp"
#include "linear_layer.hpp"
#include "relu_layer.hpp"
#include "sigmoid_layer.hpp"
#include "softmax_layer.hpp"
#include "tensor.hpp"
#include "tensor_n.hpp"
#include "cuda_tensor.hpp"
#include "cuda_tensor_n.hpp"

namespace neural {
template <typename Tensor2D_t>
struct TensorNFinderHelper
{
};

template <typename T>
struct TensorNFinderHelper<neural::Tensor<T>>
{
		using tensor_n_t = typename neural::TensorN<4, T>;
};

template <typename T>
struct TensorNFinderHelper<neural::CudaTensor<T>>
{
		using tensor_n_t = typename neural::CudaTensor4<T>;
};

template <typename Tensor2D_t>
using TensorNFinderHelper_t = typename TensorNFinderHelper<Tensor2D_t>::tensor_n_t;

template <typename Tensor2D_t>
using FlatLayer = std::variant<
    // ---- 2-D flat layers ----
    LinearLayer<Tensor2D_t>, ReLULayer<Tensor2D_t>, SigmoidLayer<Tensor2D_t>,
    SoftmaxLayer<Tensor2D_t>, DropoutLayer<Tensor2D_t>,
    BatchNorm1dLayer<Tensor2D_t>, 
		ConvolutionalBox<TensorNFinderHelper_t<Tensor2D_t>, Tensor2D_t>>;
} // namespace neural

#endif // LAYER_HPP
