#ifndef TRAINING_PROGRESS_BAR_HPP
#define TRAINING_PROGRESS_BAR_HPP

#include <cstddef>

namespace neural {

inline constexpr int kDefaultTrainingProgressBarWidth = 40;

/// Writes a single-line ``\r`` progress bar: ``[===>-----] pct%  time s/epoch  loss: ...  lr: ...``
/// \p current_step / \p total_steps drives fill level (e.g. epoch+1 and epoch_max).
void draw_training_epoch_progress_bar(
    std::size_t current_step,
    std::size_t total_steps,
    float loss_mean,
    double epoch_seconds,
    char const *prefix,
    float learning_rate,
    int bar_width = kDefaultTrainingProgressBarWidth );

} // namespace neural

#endif
