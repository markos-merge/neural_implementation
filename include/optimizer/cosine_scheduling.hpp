#ifndef COSINE_SCHEDULING_HPP
#define COSINE_SCHEDULING_HPP

#include <cstddef>

namespace neural {

/// Cosine learning-rate decay from \p lr_max down to \p lr_min over \p cosine_epochs epochs,
/// then holds \p lr_min for remaining epochs:  
/// \(\eta_t = \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})\bigl(1+\cos(\pi p)\bigr)\) with  
/// \(p = \min(\texttt{epoch}/\texttt{cosine\_epochs},\,1)\).
class CosineScheduling
{
	public:
	CosineScheduling( float lr_max, float lr_min, std::size_t cosine_epochs );

	/// \p epoch is zero-based (same convention as typical training loops).
	float learning_rate_for_epoch( std::size_t epoch ) const;

	float lr_max() const noexcept { return m_lr_max; }
	float lr_min() const noexcept { return m_lr_min; }
	std::size_t cosine_epochs() const noexcept { return m_cosine_epochs; }

	private:
	float m_lr_max{};
	float m_lr_min{};
	std::size_t m_cosine_epochs{};
};

} // namespace neural

#endif
