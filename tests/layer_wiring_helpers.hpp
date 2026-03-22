#ifndef LAYER_WIRING_HELPERS_HPP
#define LAYER_WIRING_HELPERS_HPP

namespace neural::test {

/// Wire \p LayerBase buffers: activations in→out, gradients ∂L/∂out → ∂L/∂in.
template < typename Layer, typename Tensor >
void wire_layer( Layer &layer, Tensor &in_buf, Tensor &out_buf, Tensor &grad_wrt_output,
                 Tensor &grad_wrt_input )
{
	layer.setInputOutputTensors( &in_buf, &out_buf );
	layer.setGradInputOutputTensors( &grad_wrt_output, &grad_wrt_input );
}

} // namespace neural::test

#endif
