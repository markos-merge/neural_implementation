#include "training_progress_bar.hpp"

#include <iomanip>
#include <iostream>

namespace neural {

void draw_training_epoch_progress_bar(
    std::size_t current_step,
    std::size_t total_steps,
    float loss_mean,
    double epoch_seconds,
    char const *prefix,
    float learning_rate,
    int bar_width )
{
	int const w = bar_width > 0 ? bar_width : kDefaultTrainingProgressBarWidth;
	float const pct =
	    total_steps > 0 ? static_cast<float>( current_step ) / static_cast<float>( total_steps ) : 0.f;
	int const filled = static_cast<int>( pct * static_cast<float>( w ) );
	std::cout << "\r" << prefix << "[";
	for ( int i = 0; i < w; ++i )
		std::cout << ( i < filled ? '=' : ( i == filled ? '>' : '-' ) );
	std::cout << "] " << std::fixed << std::setprecision( 1 ) << ( pct * 100.f ) << "%"
	          << "  " << std::setprecision( 3 ) << epoch_seconds << "s/epoch"
	          << "  loss: " << std::setprecision( 4 ) << loss_mean
	          << "  lr: " << std::setprecision( 6 ) << learning_rate << "    " << std::flush;
}

} // namespace neural
