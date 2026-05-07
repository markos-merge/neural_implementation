#ifndef REFACTOR_TRAINING_SCHEDULER_HPP
#define REFACTOR_TRAINING_SCHEDULER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <vector>
#include "lr_schedule.hpp"
#include "sequential_nn.hpp"
#include "tensor.hpp"

namespace neural {
namespace refactor {

// No-op batch transform — default when no augmentation is needed.
template <typename T>
struct DefaultBatchTransform {
    void operator()( neural::Tensor<T> &, neural::Tensor<T> & ) const {}
};

// Controls how an optimizer receives data and how the learning rate changes.
//
// Responsibilities:
//   - Batches the dataset each epoch from per-sample Tensor<T> vectors.
//   - Optionally shuffles the sample order before each epoch.
//   - Applies m_lr_schedule at the start of each epoch to set opt.m_learning_rate.
//   - After assembling each flat batch, calls m_batch_transform(batch_x, batch_y)
//     (non-owning Tensor views) — use this for augmentation, normalization, etc.
//   - Calls opt.step() for each batch and reports progress via an optional callback.
//
// The optimizer is a pure gradient-update engine; the scheduler drives it.
//
// Usage (with augmentation):
//   using Opt   = MomentumSGDOptimizer<float, Cuda, CrossEntropyLoss<float, Cuda>>;
//   using Sched = TrainingScheduler<float, Cuda, Opt,
//                                   CosineAnnealingLR<float>,
//                                   Cifar10AugmentBatchOp>;
//   Opt opt(0.0f, 0.9f);
//   opt.m_l2_regularizer = 1e-4f;
//
//   Sched sched;
//   sched.m_epochs          = 30;
//   sched.m_batch_size      = 512;
//   sched.m_lr_schedule     = { .m_lr_max = 0.1f, .m_lr_min = 1e-4f };
//   sched.m_batch_transform = std::move(augment_op);
//   sched.train(opt, nn, train_images, train_labels,
//       [](auto epoch, auto tot_e, auto batch, auto tot_b, auto loss, auto lr) {
//           return false;  // return true to stop early
//       });

template <typename T,
          typename Device,
          typename Optimizer,
          typename LRSchedule      = ConstantLR<T>,
          typename BatchTransform  = DefaultBatchTransform<T>>
class TrainingScheduler
{
public:
    std::size_t    m_epochs          = 100;
    std::size_t    m_batch_size      = 32;
    bool           m_shuffle         = true;
    std::uint32_t  m_seed            = 42;
    LRSchedule     m_lr_schedule{};
    BatchTransform m_batch_transform{};

    // Called after every batch.  Return true to abort training early.
    using ProgressCallback = std::function<bool( std::size_t epoch,
                                                  std::size_t total_epochs,
                                                  std::size_t batch,
                                                  std::size_t total_batches,
                                                  T           loss,
                                                  T           lr )>;

    void train( Optimizer&                            opt,
                SequentialNN2<T, Device>&             nn,
                std::vector<neural::Tensor<T>> const& inputs,
                std::vector<neural::Tensor<T>> const& targets,
                ProgressCallback                      cb = nullptr );

private:
    static void fill_batch( std::vector<neural::Tensor<T>> const& inputs,
                             std::vector<neural::Tensor<T>> const& targets,
                             std::vector<std::size_t> const&       order,
                             std::size_t                           start,
                             std::size_t                           count,
                             std::vector<T>&                       batch_x,
                             std::vector<T>&                       batch_y );
};

// ---------------------------------------------------------------------------

template <typename T, typename Device, typename Optimizer, typename LRSchedule, typename BatchTransform>
void TrainingScheduler<T, Device, Optimizer, LRSchedule, BatchTransform>::fill_batch(
    std::vector<neural::Tensor<T>> const& inputs,
    std::vector<neural::Tensor<T>> const& targets,
    std::vector<std::size_t> const&       order,
    std::size_t                           start,
    std::size_t                           count,
    std::vector<T>&                       batch_x,
    std::vector<T>&                       batch_y )
{
    std::size_t const in_cols  = inputs[0].cols();
    std::size_t const out_cols = targets[0].cols();
    batch_x.resize( count * in_cols );
    batch_y.resize( count * out_cols );
    for ( std::size_t i = 0; i < count; ++i ) {
        std::size_t const idx = order[start + i];
        std::copy_n( inputs[idx].data(),  in_cols,  batch_x.data() + i * in_cols );
        std::copy_n( targets[idx].data(), out_cols, batch_y.data() + i * out_cols );
    }
}

template <typename T, typename Device, typename Optimizer, typename LRSchedule, typename BatchTransform>
void TrainingScheduler<T, Device, Optimizer, LRSchedule, BatchTransform>::train(
    Optimizer&                            opt,
    SequentialNN2<T, Device>&             nn,
    std::vector<neural::Tensor<T>> const& inputs,
    std::vector<neural::Tensor<T>> const& targets,
    ProgressCallback                      cb )
{
    std::size_t const N                 = inputs.size();
    std::size_t const in_cols           = inputs[0].cols();
    std::size_t const out_cols          = targets[0].cols();
    std::size_t const batches_per_epoch = N / m_batch_size;

    std::vector<std::size_t> order( N );
    std::iota( order.begin(), order.end(), 0 );
    std::mt19937 rng( m_seed );

    std::vector<T> batch_x, batch_y;

    for ( std::size_t epoch = 0; epoch < m_epochs; ++epoch ) {
        T const lr          = m_lr_schedule( epoch, m_epochs );
        opt.m_learning_rate = lr;

        if ( m_shuffle )
            std::shuffle( order.begin(), order.end(), rng );

        for ( std::size_t bi = 0; bi < batches_per_epoch; ++bi ) {
            std::size_t const start = bi * m_batch_size;
            std::size_t const count = std::min( m_batch_size, N - start );

            fill_batch( inputs, targets, order, start, count, batch_x, batch_y );

            neural::Tensor<T> tx( batch_x.data(), count, in_cols,  false );
            neural::Tensor<T> ty( batch_y.data(), count, out_cols, false );
            m_batch_transform( tx, ty );

            T const loss = opt.step( nn, batch_x.data(), count, in_cols, batch_y.data() );

            if ( cb && cb( epoch, m_epochs, bi, batches_per_epoch, loss, lr ) )
                return;
        }
    }
}

} // namespace refactor
} // namespace neural

#endif // REFACTOR_TRAINING_SCHEDULER_HPP
