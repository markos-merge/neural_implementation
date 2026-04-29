#ifndef TRAINING_EPOCH_PROGRESS_REPORTER_HPP
#define TRAINING_EPOCH_PROGRESS_REPORTER_HPP

#include <chrono>
#include <cstddef>

namespace neural {

/// Owns per-epoch loss accumulation and wall-clock timing between epoch boundaries, and draws the
/// epoch progress bar (\ref draw_training_epoch_progress_bar) when the last batch of an epoch completes.
class TrainingEpochProgressReporter
{
	public:
	TrainingEpochProgressReporter();

	/// Adds \p batch_loss for batch \p batch_idx. Returns \c false until the last batch of the epoch
	/// (\p batch_idx + 1 == \p batch_max): then computes mean loss and elapsed seconds since the previous
	/// epoch boundary, redraws the progress bar (with \p learning_rate printed), resets accumulation state,
	/// returns \c true.
	bool consume_batch(
	    float batch_loss,
	    std::size_t batch_idx,
	    std::size_t batch_max,
	    std::size_t epoch_index_zero_based,
	    std::size_t epoch_max,
	    float learning_rate );

	private:
	std::chrono::steady_clock::time_point m_last_epoch_end;
	float m_loss_sum{};
};

} // namespace neural

#endif
