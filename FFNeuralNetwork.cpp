#include "FFNeuralNetwork.h"

FFNeuralNetwork::FFNeuralNetwork() {
    layers = new NeuronLayer[3];

    layers[0].init(2, 1);
    layers[1].init(2, 1);
    layers[2].init(1, 0);

    layers[0].connectTo(layers[1]);
    layers[1].connectTo(layers[2]);

    layerCount = 3;

    activationFuncion = sigmoid;
    activationDerivative = sigmoidDerivative;
}

FFNeuralNetwork::FFNeuralNetwork(const std::string& filename) {

}

FFNeuralNetwork::FFNeuralNetwork(size_t inputNeurons, size_t numHidden, size_t neuronPerHidden, size_t outputNeurons) {

}

FFNeuralNetwork::FFNeuralNetwork(size_t inputNeurons, size_t numHidden, size_t* hiddenSizes, size_t outputNeurons) {

}

FFNeuralNetwork::~FFNeuralNetwork() {
    delete[] layers;
}

void FFNeuralNetwork::init(size_t inputNeurons, size_t numHidden, size_t neuronPerHidden, size_t outputNeurons) {

}

void FFNeuralNetwork::init(size_t inputNeurons, size_t numHidden, size_t* hiddenSizes, size_t outputNeurons) {

}

void FFNeuralNetwork::setFunctions(std::function<float(float)> activFunc, std::function<float(float)> deriv) {

}

void FFNeuralNetwork::setInputs(const float* vals, size_t arrSize) {

}

void FFNeuralNetwork::setInputs(const std::vector<float>& vals) {

}

void FFNeuralNetwork::propagateForwards() {

}

std::vector<float> FFNeuralNetwork::getOutputs() {

}

std::vector<float> FFNeuralNetwork::processData(const float* vals, size_t arrSize) {

}

std::vector<float> FFNeuralNetwork::processData(const std::vector<float>& vals) {

}

void FFNeuralNetwork::train(const float* inputData, size_t inputSize, const float* expected, size_t expectedSize, size_t numEpochs, float learningRate, float momentum) {

}

void FFNeuralNetwork::train(const std::vector<float>& inputData, const std::vector<float>& expected, size_t numEpochs, float learningRate, float momentum) {

}

std::vector<float> FFNeuralNetwork::calculateOutputError(const float* expected) {

}

std::vector<float> FFNeuralNetwork::calculateOutputError(const std::vector<float>& expected) {

}

std::vector<float> FFNeuralNetwork::calculateHiddenError(NeuronLayer& nl) {

}

size_t FFNeuralNetwork::size() const {
    return layerCount;
}

NeuronLayer& FFNeuralNetwork::operator[](size_t index) {

}

const NeuronLayer& FFNeuralNetwork::operator[](size_t index) const {

}

std::ostream& operator<<(std::ostream& stream, const FFNeuralNetwork& ffnn) {
    stream << "Network {\n";

    stream << "\tlayerCount = " << ffnn.layerCount << ";\n";
    stream << "\tinputSize = " << ffnn.layers[0].size() << ";\n";
    stream << "\toutputSize = " << ffnn.layers[ffnn.size() - 1].size() << ";\n";

    for(size_t i = 0; i < ffnn.layerCount; ++i) {
        stream << ffnn.layers[i] << "\n";
    }

    stream << "};\n";

    return stream;
}
