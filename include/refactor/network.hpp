#ifndef REFACTOR_NETWORK_HPP
#define REFACTOR_NETWORK_HPP

#include "device.hpp"
#include "dtype.hpp"
#include "sequential_nn.hpp"

#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace neural {
namespace refactor {

template <typename T, typename Device>
class Network
{
public:
	void forward( T const *data, std::vector<std::size_t> const &shape )
	{
		forwardImpl( data, shape );
	}
	void backward( T const *grad_output ) { backwardImpl( grad_output ); }

	virtual DType       dtype() const = 0;
	virtual std::size_t outputSize() const = 0;
	virtual ~Network()  = default;

protected:
	virtual void forwardImpl( T const *data, std::vector<std::size_t> const &shape ) = 0;
	virtual void backwardImpl( T const *grad_output )                           = 0;
};

template <typename T, typename Device = Cpu>
class SequentialNetwork : public Network<T, Device>
{
public:
	template <typename F>
	void forEachLayer( F &&f )
	{
		m_nn.forEachLayer( std::forward<F>( f ) );
	}

	DType dtype() const override { return dtype_of<T>; }

	std::size_t outputSize() const override { return m_nn.outputSize(); }

	SequentialNN2<T, Device> &nn() noexcept { return m_nn; }
	SequentialNN2<T, Device> const &nn() const noexcept { return m_nn; }

protected:
	void forwardImpl( T const *data, std::vector<std::size_t> const &shape ) override
	{
		if ( shape.size() < 2 ) {
			throw std::invalid_argument(
			    "SequentialNetwork::forward: shape must have at least batch and feature dims" );
		}
		std::size_t const N = shape[0];
		std::size_t const in_flat = std::accumulate( shape.begin() + 1, shape.end(),
		    std::size_t{ 1 }, std::multiplies<std::size_t>() );
		m_nn.forward( data, N, in_flat );
	}

	void backwardImpl( T const *grad_output ) override { m_nn.backward( grad_output ); }

private:
	SequentialNN2<T, Device> m_nn;
};

} // namespace refactor
} // namespace neural

#include "nn_io.hpp"

namespace neural {
namespace refactor {

template <typename T, typename Device>
void save( SequentialNetwork<T, Device> &sn, std::string const &path )
{
	save( sn.nn(), path );
}

template <typename T, typename Device>
void load( SequentialNetwork<T, Device> &sn, std::string const &path )
{
	load( sn.nn(), path );
}

} // namespace refactor
} // namespace neural

#endif // REFACTOR_NETWORK_HPP
