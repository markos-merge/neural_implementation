#ifndef REFACTOR_LATCH_BASE_HPP
#define REFACTOR_LATCH_BASE_HPP

#include <cstddef>
#include <vector>
#include "device.hpp"

namespace neural {
namespace refactor {

template <typename T, typename Device = Cpu>
class LatchBase
{
	public:
		virtual ~LatchBase() = default;

		virtual T *fwdData() = 0;
		virtual T *bwdData() = 0;
		virtual std::vector<std::size_t> const &shape() const = 0;
		virtual void resize( std::vector<std::size_t> const &s ) = 0;
};

} // namespace refactor
} // namespace neural

#endif // REFACTOR_LATCH_BASE_HPP
