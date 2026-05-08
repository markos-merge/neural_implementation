#ifndef REFACTOR_LR_SCHEDULE_HPP
#define REFACTOR_LR_SCHEDULE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace neural {
namespace refactor {

// All schedules satisfy: T operator()(std::size_t epoch, std::size_t total_epochs) const
// The scheduler calls schedule(epoch, total) and writes the result into opt.m_learning_rate.

// ── Constant
// ──────────────────────────────────────────────────────────────────
template <typename T>
struct ConstantLR
{
		T m_lr = T( 0.01 );
		T operator()( std::size_t /*epoch*/, std::size_t /*total*/ ) const
		{
			return m_lr;
		}
};

// ── Step decay
// ──────────────────────────────────────────────────────────────── lr =
// lr_initial * gamma ^ (epoch / step_size)  (integer division)
template <typename T>
struct StepLR
{
		T m_lr_initial;
		T m_gamma = T( 0.1 );
		std::size_t m_step_size = 10;

		T operator()( std::size_t epoch, std::size_t /*total*/ ) const
		{
			return m_lr_initial *
			       std::pow( m_gamma, static_cast<T>( epoch / m_step_size ) );
		}
};

// ── Cosine annealing
// ────────────────────────────────────────────────────────── lr = lr_min +
// 0.5*(lr_max - lr_min)*(1 + cos(pi * epoch/total))
template <typename T>
struct CosineAnnealingLR
{
		T m_lr_max;
		T m_lr_min = T( 0 );

		T operator()( std::size_t epoch, std::size_t total ) const
		{
			T const p =
			    total > 0
			        ? std::min( static_cast<T>( epoch ) / static_cast<T>( total ),
			                    T( 1 ) )
			        : T( 1 );
			return m_lr_min + T( 0.5 ) * ( m_lr_max - m_lr_min ) *
			                      ( T( 1 ) + std::cos( static_cast<T>( M_PI ) * p ) );
		}
};

// ── Exponential decay
// ───────────────────────────────────────────────────────── lr = lr_initial *
// gamma^epoch
template <typename T>
struct ExponentialLR
{
		T m_lr_initial;
		T m_gamma = T( 0.95 );

		T operator()( std::size_t epoch, std::size_t /*total*/ ) const
		{
			return m_lr_initial * std::pow( m_gamma, static_cast<T>( epoch ) );
		}
};

} // namespace refactor
} // namespace neural

#endif // REFACTOR_LR_SCHEDULE_HPP
