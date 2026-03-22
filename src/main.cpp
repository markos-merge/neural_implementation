#include "cifar_demo.hpp"
#include "mnist_demo.hpp"
#include <iostream>

int main()
{
	std::cout << "Neural Network Initialized!" << std::endl;
	// run_mnist_demo();
	run_cifar10_demo();
	return 0;
}
