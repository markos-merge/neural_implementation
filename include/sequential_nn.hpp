#ifndef SEQUENTIAL_NN_HPP
#define SEQUENTIAL_NN_HPP
#include "tensor_like.hpp"
#include "mse_loss.hpp"

namespace neural {

template <typename Tensor, typename Loss, typename ... Layers>
class SequentialNN
{
	static_assert( TensorLike<Tensor>, "Tensor must be a TensorLike type" );
	public:

		explicit SequentialNN( Layers... layers );

		Tensor forward( Tensor const &input );
		Tensor backward( Tensor const &grad_output );
		
		typename Tensor::value_type trainStep( Tensor const &input, Tensor const &target );
		
		template < typename UnaryOp >
		void forEachLayer( UnaryOp &&op );

	private:
		std::tuple< Layers... > m_layers;
		Loss m_loss_fn;
};

template < typename Tensor, typename Loss, typename ...Layers >
SequentialNN<Tensor, Loss, Layers...>::SequentialNN( Layers... layers )
	:m_layers( layers... )
{

}

template < typename Tensor, typename Layer >
Tensor forward_impl( Tensor const &input, Layer &layer )
{
	return layer.forward( input );
}

template < typename Tensor, typename Layer, typename ...Layers >
Tensor forward_impl( Tensor const &input, Layer &layer, Layers &...layers )
{
	return forward_impl( layer.forward( input ), layers... );
}

template < typename Tensor, typename Loss, typename ...Layers >
Tensor SequentialNN<Tensor, Loss, Layers...>::forward( Tensor const &input )
{
	return std::apply( [&input]( auto &&...args ) { return forward_impl( input, args... ); }, m_layers );
}

template < typename Tensor, typename Layer >
Tensor backward_impl( Tensor const &grad_output, Layer &layer )
{
	return layer.backward( grad_output );
}

template < typename Tensor, typename Layer, typename ...Layers >
Tensor backward_impl( Tensor const &grad_output, Layer &layer, Layers &...layers )
{
	return layer.backward( backward_impl( grad_output, layers... ) );
}

template < typename Tensor, typename Loss, typename ...Layers >
Tensor SequentialNN<Tensor, Loss, Layers...>::backward( Tensor const &grad_output )
{
	return std::apply( [&grad_output]( auto &&...args ) { return backward_impl( grad_output, args... ); }, m_layers );
}

template< typename Tensor, typename Loss, typename ...Layers >
typename Tensor::value_type SequentialNN<Tensor, Loss, Layers...>::trainStep( Tensor const &input, Tensor const &target )
{
	Tensor output = forward( input );
	auto loss = m_loss_fn.forward( output, target );
	backward( m_loss_fn.backward() );
	return loss;
}

template < typename Tensor, typename Loss, typename ...Layers >
template < typename UnaryOp >
void SequentialNN<Tensor, Loss, Layers...>::forEachLayer( UnaryOp &&op )
{
	std::apply( [&op]( auto &&...args ) { (op( args ), ...); }, m_layers );
}

template <typename Tensor, typename ... Layers>
using SequentialNNMSE = SequentialNN<Tensor, MSELoss<Tensor>, Layers...>;

}
#endif