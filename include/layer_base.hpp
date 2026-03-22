#ifndef LAYER_BASE_HPP
#define LAYER_BASE_HPP

namespace neural {

template <typename Tensor>
class LayerBase
{
	public:
		virtual ~LayerBase() = default;

		void setInputOutputTensors( Tensor *input, Tensor *output );
		void setGradInputOutputTensors( Tensor *grad_input, Tensor *grad_output );

		Tensor *getInput();
		Tensor *getOutput();
		Tensor *getGradInput();
		Tensor *getGradOutput();

	private:
		Tensor *m_input = nullptr;
		Tensor *m_output = nullptr;
		Tensor *m_grad_input = nullptr;
		Tensor *m_grad_output = nullptr;
};

template <typename Tensor>
void LayerBase<Tensor>::setInputOutputTensors( Tensor *input, Tensor *output )
{
	m_input = input;
	m_output = output;
}

template <typename Tensor>
void LayerBase<Tensor>::setGradInputOutputTensors( Tensor *grad_input, Tensor *grad_output )
{
	m_grad_input = grad_input;
	m_grad_output = grad_output;
}

template <typename Tensor>
Tensor *LayerBase<Tensor>::getInput()
{
	return m_input;
}

template <typename Tensor>
Tensor *LayerBase<Tensor>::getOutput()
{
	return m_output;
}

template <typename Tensor>
Tensor *LayerBase<Tensor>::getGradInput()
{
	return m_grad_input;
}

template <typename Tensor>
Tensor *LayerBase<Tensor>::getGradOutput()
{
	return m_grad_output;
}

} // namespace neural

#endif