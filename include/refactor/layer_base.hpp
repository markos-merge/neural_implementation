#ifndef REFACTOR_LAYER_BASE_HPP
#define REFACTOR_LAYER_BASE_HPP

#include "latch_base.hpp"

namespace neural {
namespace refactor {

template <typename T, typename Device = Cpu>
class LayerBase
{
public:
	void setInputOutputLatches( LatchBase<T, Device> *input, LatchBase<T, Device> *output );
	void setGradLatches( LatchBase<T, Device> *grad_in, LatchBase<T, Device> *grad_out );

	LatchBase<T, Device> *getInput();
	LatchBase<T, Device> *getOutput();
	LatchBase<T, Device> *getGradInput();
	LatchBase<T, Device> *getGradOutput();

private:
	LatchBase<T, Device> *m_input   = nullptr;
	LatchBase<T, Device> *m_output  = nullptr;
	LatchBase<T, Device> *m_grad_in  = nullptr;
	LatchBase<T, Device> *m_grad_out = nullptr;
};

template <typename T, typename Device>
void LayerBase<T, Device>::setInputOutputLatches( LatchBase<T, Device> *input, LatchBase<T, Device> *output )
{
	m_input  = input;
	m_output = output;
}

template <typename T, typename Device>
void LayerBase<T, Device>::setGradLatches( LatchBase<T, Device> *grad_in, LatchBase<T, Device> *grad_out )
{
	m_grad_in  = grad_in;
	m_grad_out = grad_out;
}

template <typename T, typename Device>
LatchBase<T, Device> *LayerBase<T, Device>::getInput()
{
	return m_input;
}

template <typename T, typename Device>
LatchBase<T, Device> *LayerBase<T, Device>::getOutput()
{
	return m_output;
}

template <typename T, typename Device>
LatchBase<T, Device> *LayerBase<T, Device>::getGradInput()
{
	return m_grad_in;
}

template <typename T, typename Device>
LatchBase<T, Device> *LayerBase<T, Device>::getGradOutput()
{
	return m_grad_out;
}

} // namespace refactor
} // namespace neural

#endif // REFACTOR_LAYER_BASE_HPP
