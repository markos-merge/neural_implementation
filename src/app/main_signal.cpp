#include "main.hpp"

#include <csignal>

namespace {

std::sig_atomic_t volatile g_training_interrupt{};

extern "C" void neural_main_sigint_handler( int signal_number )
{
	g_training_interrupt = static_cast<std::sig_atomic_t>( signal_number );
}

} // namespace

namespace neural {

MainSignal &MainSignal::instance()
{
	static MainSignal inst;
	return inst;
}

void MainSignal::reset_interrupt_flag()
{
	g_training_interrupt = 0;
}

bool MainSignal::interrupt_requested() const noexcept
{
	return g_training_interrupt != 0;
}

void MainSignal::install_sigint_handler()
{
	std::signal( SIGINT, neural_main_sigint_handler );
}

} // namespace neural
