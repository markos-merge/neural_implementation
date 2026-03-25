#ifndef NEURAL_IMPL_TENSOR_N_HPP
#define NEURAL_IMPL_TENSOR_N_HPP

#include <cstddef>
#include <vector>
#include <iterator>

namespace neural {

template <typename T = float>
class Tensor;

// Rank-2 slices return Tensor<T> for interoperability with existing layers (see docs/1.tensor/TENSOR_N.md).
template <std::size_t rank, typename T = float>
class TensorN
{
	public:
		using value_type = T;
		static constexpr std::size_t rank_v = rank;
	
		TensorN() noexcept = default;
		TensorN( std::array<std::size_t, rank> shape );
		TensorN( std::array< std::size_t, rank > shape, value_type value );
		template <std::random_access_iterator It>
		TensorN( std::array<std::size_t, rank> shape, It begin, It end );
		explicit TensorN( std::vector< value_type > const &data, std::array< std::size_t, rank > const &shape ) noexcept;
		
		TensorN( TensorN const &other ) noexcept;
		TensorN( TensorN &&other ) noexcept;
		TensorN &operator=( TensorN const &other ) noexcept;
		TensorN &operator=( TensorN &&other ) noexcept;
		~TensorN() = default;
		
		std::array<std::size_t, rank> shape() const noexcept;
		std::size_t size() const noexcept;
		std::size_t ndim() const noexcept;
	
		value_type operator()( std::array< std::size_t, rank > const indices );
	
		void assign( std::vector< value_type > const &data );

		template< std::size_t other_rank >
		void assign( std::array< std::size_t, other_rank > const &from_slice, std::array< std::size_t, rank - other_rank > const &to_slice, TensorN< rank - other_rank, T > const &tensor );
};

} // namespace neural

#endif // NEURAL_IMPL_TENSOR_N_HPP
