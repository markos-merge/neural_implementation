#ifndef NEURAL_MAIN_HPP
#define NEURAL_MAIN_HPP

namespace neural {

/// Singleton facade for a process-wide training interrupt flag set by \c SIGINT (cooperative shutdown).
/// Storage lives in \ref src/app/main_signal.cpp (async-signal-safe handler).
class MainSignal
{
	public:
	static MainSignal &instance();

	void reset_interrupt_flag();
	bool interrupt_requested() const noexcept;

	/// Registers \c SIGINT to record interrupt; pair with \ref reset_interrupt_flag before training.
	void install_sigint_handler();

	private:
	MainSignal() = default;
};

} // namespace neural

#endif
