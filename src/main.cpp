#include "cifar_demo.hpp"
#include "conv_demo_momentum.hpp"
#include "mnist_demo.hpp"
#include <iostream>
#include <omp.h>
#include "conv_demo.hpp"

int main()
{
#ifdef _OPENMP
	std::cout << "OpenMP is enabled and the number of threads is " << omp_get_max_threads() << std::endl;
#endif
	std::cout << "Neural Network Initialized!" << std::endl;
	// run_mnist_demo();
	// run_cifar10_demo();
	// run_conv_demo();
	run_conv_demo_momentum();
	return 0;
}
