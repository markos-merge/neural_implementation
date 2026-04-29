#include "cosine_scheduling.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace neural {

CosineScheduling::CosineScheduling( float lr_max, float lr_min, std::size_t cosine_epochs )
    : m_lr_max( lr_max )
    , m_lr_min( lr_min )
    , m_cosine_epochs( cosine_epochs )
{}

float CosineScheduling::learning_rate_for_epoch( std::size_t epoch ) const
{
	float progress = 1.f;
	if ( m_cosine_epochs > 0 ) {
		progress = std::min(
		    static_cast<float>( epoch ) / static_cast<float>( m_cosine_epochs ), 1.f );
	}
	return m_lr_min + 0.5f * ( m_lr_max - m_lr_min )
	       * ( 1.f + std::cos( std::numbers::pi_v<float> * progress ) );
}

} // namespace neural
