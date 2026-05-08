#ifndef CONV_DEMO_REFACTOR_CKPT_HPP
#define CONV_DEMO_REFACTOR_CKPT_HPP

/// CIFAR-10 CUDA conv demo (same topology as \c run_conv_demo_refactor): 500 epochs, batch 128,
/// checkpoint every 20 epochs under \c nn_checkpoints/, then scores each checkpoint on train+test.
void run_conv_demo_refactor_ckpt();

#endif
