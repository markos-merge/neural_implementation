#ifndef REFACTOR_LAYER_HPP
#define REFACTOR_LAYER_HPP

#include <variant>
#include "batch_norm_layer.hpp"
#include "convolutional_layer.hpp"
#include "dropout_layer.hpp"
#include "linear_layer.hpp"
#include "max_pool_layer.hpp"
#include "relu_layer.hpp"
#include "sigmoid_layer.hpp"
#include "softmax_layer.hpp"

namespace neural {
namespace refactor {

// Single variant covering all layer types.
// ConvolutionalLayer is the 4D entry point — use the {C,H,W} constructor overload.
// LinearLayer exits 4D implicitly by computing in_features = product(shape[1:]) on first forward.
template <typename T, typename Device = Cpu>
using Layer = std::variant<
    LinearLayer<T, Device>,
    ReLULayer<T, Device>,
    SigmoidLayer<T, Device>,
    SoftmaxLayer<T, Device>,
    DropoutLayer<T, Device>,
    BatchNormLayer<T, Device>,
    ConvolutionalLayer<T, Device>,
    MaxPoolLayer<T, Device>
>;

} // namespace refactor
} // namespace neural

#endif // REFACTOR_LAYER_HPP
