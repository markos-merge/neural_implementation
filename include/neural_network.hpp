#include <vector>

class NeuralNetwork {
	public:
		NeuralNetwork( float learning_rate = 0.01f );

		[[nodiscard]] bool train( std::vector<std::vector<float>> const &data, size_t epochs = 100 );

		void predict( std::vector<float> const &input );

	private:
		float learning_rate;
		std::vector<std::vector<float>> weights;
		std::vector<float> biases;
};