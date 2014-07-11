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

FFNeuralNetwork::FFNeuralNetwork(size_t inputNeurons, size_t inputBias, size_t numHidden, size_t neuronPerHidden, size_t biasPerHidden, size_t outputNeurons) {
    layerCount = numHidden + 2;

    layers = new NeuronLayer[numHidden + 2];

    layers[0].init(inputNeurons, inputBias);

    for(size_t i = 1; i < numHidden + 1; ++i) {
        layers[i].init(neuronPerHidden, biasPerHidden);
    }

    layers[numHidden + 1].init(outputNeurons, 0);

    for(size_t i = 0; i < layerCount - 1; ++i) {
        layers[i].connectTo(layers[i+1]);
    }

    activationFuncion = sigmoid;
    activationDerivative = sigmoidDerivative;
}

FFNeuralNetwork::FFNeuralNetwork(size_t inputNeurons, size_t inputBias, size_t numHidden, size_t* hiddenSizes, size_t* biasAmounts, size_t outputNeurons) {

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
    if(index < layerCount) {
        return layers[index];
    } else {
        throw std::out_of_range("There are fewer layers than the value of the given index + 1.");
    }
}

const NeuronLayer& FFNeuralNetwork::operator[](size_t index) const {
    if(index < layerCount) {
        return layers[index];
    } else {
        throw std::out_of_range("There are fewer layers than the value of the given index + 1.");
    }
}

std::ostream& operator<<(std::ostream& stream, const FFNeuralNetwork& ffnn) {
    stream << "Network {\n";

    stream << "\tlayerCount = " << ffnn.layerCount << ";\n";
    stream << "\tinputSize = " << ffnn[0].size() << ";\n";
    stream << "\toutputSize = " << ffnn[ffnn.size() - 1].size() << ";\n";

    stream << "\thiddenSizes = [\n";

    for(size_t i = 1; i < ffnn.layerCount - 1; ++i) {
        stream << "\t\t" << ffnn[i].size();

        if(i < ffnn.layerCount - 2) {
            stream << ",\n";
        } else {
            stream << "\n";
        }
    }

    stream << "\t];\n\n";

    for(size_t i = 0; i < ffnn.size(); ++i) {
        stream << ffnn[i] << "\n";
    }

    stream << "};\n";

    return stream;
}

//------------------------------------------------------------------------\\

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x) );
}

float sigmoidDerivative(float x) {
    float sigRes = sigmoid(x);
    return sigRes * (1.0f - sigRes);
}

FFNeuralNetwork loadNN(std::string filename) {
    std::fstream file;
    std::stringstream valueReader;

    size_t numLayers, inputSize, outputSize;
    size_t* hiddenSizes;

    float* weights;

    file.open(filename.c_str(), std::ios_base::in);

    std::string line;


}
