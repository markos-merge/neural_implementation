#include "cifar_demo.hpp"
#include "conv_demo_cpu_small.hpp"
#include "conv_demo_cuda.hpp"
#include "conv_demo_cuda_small.hpp"
#include "conv_demo_momentum.hpp"
#include "main.hpp"
#include "mnist_demo.hpp"
#include <iostream>
#include <omp.h>
#include "conv_demo.hpp"

int main()
{
	neural::MainSignal::instance().reset_interrupt_flag();

#ifdef _OPENMP
	std::cout << "OpenMP is enabled and the number of threads is " << omp_get_max_threads() << std::endl;
#endif
	std::cout << "Neural Network Initialized!" << std::endl;
	// run_mnist_demo();
	// run_cifar10_demo();
	// run_conv_demo();
	// run_conv_demo_momentum();
	// run_conv_demo_cpu_small(); // TensorN im2col (CPU); see header for vs. CUDA geometry
	run_conv_demo_cuda_small();
	// run_conv_demo_cuda();
	return 0;
}
