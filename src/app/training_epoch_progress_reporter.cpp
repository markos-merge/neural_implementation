#include "training_epoch_progress_reporter.hpp"

#include "training_progress_bar.hpp"

#include <chrono>
#include <string>

namespace neural {

TrainingEpochProgressReporter::TrainingEpochProgressReporter()
    : m_last_epoch_end( std::chrono::steady_clock::now() )
{}

bool TrainingEpochProgressReporter::consume_batch(
    float batch_loss,
    std::size_t batch_idx,
    std::size_t batch_max,
    std::size_t epoch_index_zero_based,
    std::size_t epoch_max,
    float learning_rate )
{
	m_loss_sum += batch_loss;
	if ( batch_idx + 1 < batch_max )
		return false;

	float const mean_loss = m_loss_sum / static_cast<float>( batch_max );
	m_loss_sum          = 0.f;

	auto const now = std::chrono::steady_clock::now();
	double const elapsed =
	    std::chrono::duration<double>( now - m_last_epoch_end ).count();
	m_last_epoch_end = now;

	std::string const prefix =
	    "Epoch " + std::to_string( epoch_index_zero_based + 1 ) + "/" +
	    std::to_string( epoch_max ) + " ";
	draw_training_epoch_progress_bar(
	    epoch_index_zero_based + 1, epoch_max, mean_loss, elapsed, prefix.c_str(), learning_rate );
	return true;
}

} // namespace neural
