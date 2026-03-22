#ifndef LATCH_LAYER_HPP
#define LATCH_LAYER_HPP

#include <cstddef>

namespace neural {

template <typename Tensor>
class LatchLayer
{
	public:
		LatchLayer( std::size_t rows, std::size_t cols );

		Tensor *input();
		Tensor *grads();

		Tensor *forward();
		Tensor *backward();

		Tensor *getInput();
		Tensor *getOutput();
		Tensor *getGradInput();
		Tensor *getGradOutput();

	private:
		Tensor m_input;
		Tensor m_grads;
};

template <typename Tensor>
LatchLayer<Tensor>::LatchLayer( std::size_t rows, std::size_t cols )
	: m_input( rows, cols )
	, m_grads( rows, cols )
{
}

template <typename Tensor>
Tensor *LatchLayer<Tensor>::input()
{
	return &m_input;
}

template <typename Tensor>
Tensor *LatchLayer<Tensor>::grads()
{
	return &m_grads;
}

template <typename Tensor>
Tensor *LatchLayer<Tensor>::forward()
{
	return input();
}

template <typename Tensor>
Tensor *LatchLayer<Tensor>::backward()
{
	return grads();
}

template <typename Tensor>
Tensor *LatchLayer<Tensor>::getInput()
{
	return input();
}

template <typename Tensor>
Tensor *LatchLayer<Tensor>::getOutput()
{
	return input();
}

template <typename Tensor>
Tensor *LatchLayer<Tensor>::getGradInput()
{
	return grads();
}

template <typename Tensor>
Tensor *LatchLayer<Tensor>::getGradOutput()
{
	return grads();
}

} // namespace neural

#endif
