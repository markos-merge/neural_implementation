#ifndef TERMINAL_LATCH_LAYER_HPP
#define TERMINAL_LATCH_LAYER_HPP

#include "layer_base.hpp"
#include "tensor_like.hpp"

namespace neural {
template <typename Tensor>
class TerminalLatchLayer : public LayerBase< Tensor >
{
	static_assert( TensorLike<Tensor>, "Tensor must be a TensorLike type" );
	public:
		TerminalLatchLayer() = default;

		TerminalLatchLayer( Tensor *input, Tensor *grads );

		void rebind( Tensor *input, Tensor *grads );

		Tensor *forward();
		Tensor *backward();
};

template <typename Tensor>
TerminalLatchLayer<Tensor>::TerminalLatchLayer( Tensor *input, Tensor *grads )
{
	rebind( input, grads );
}

template <typename Tensor>
void TerminalLatchLayer<Tensor>::rebind( Tensor *input, Tensor *grads )
{
	this->setInputOutputTensors( input, input );
	this->setGradInputOutputTensors( grads, grads );
}

template <typename Tensor>
Tensor *TerminalLatchLayer<Tensor>::forward()
{
	return this->getOutput();
}

template <typename Tensor>
Tensor *TerminalLatchLayer<Tensor>::backward()
{
	return this->getGradOutput();
}

} // namespace neural

#endif
