#include <iostream>
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(float learning_rate = 0.01f);

    [[nodiscard]] bool train(const std::vector<std::vector<float>>& data, size_t epochs = 100);

    void predict(const std::vector<float>& input);

private:
    float learning_rate;
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
};