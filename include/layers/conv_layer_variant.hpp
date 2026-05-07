#ifndef CONV_LAYER_VARIANT_HPP
#define CONV_LAYER_VARIANT_HPP

#include <variant>
#include "batch_norm_layer.hpp"
#include "convolutional_layer.hpp"
#include "dropout_layer.hpp"
#include "max_pool_layer.hpp"
#include "relu_layer.hpp"

namespace neural {

/// Variant of the layers that can live inside a ConvolutionalBox (all rank-4 NCHW).
template <typename TensorN_t>
using ConvLayer = std::variant<
    ConvolutionalLayer<TensorN_t>,
    MaxPoolLayer<TensorN_t>,
    BatchNormLayer<TensorN_t>,
    ReLULayer<TensorN_t>,
    DropoutLayer<TensorN_t>
>;

} // namespace neural

#endif // CONV_LAYER_VARIANT_HPP
