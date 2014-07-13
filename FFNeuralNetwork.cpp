#include "FFNeuralNetwork.h"

FFNeuralNetwork::FFNeuralNetwork() {
    layers = new NeuronLayer[3];

    layers[0].init(2, 1);
    layers[1].init(2, 1);
    layers[2].init(1, 0);

    layers[0].connectTo(layers[1]);
    layers[1].connectTo(layers[2]);

    layerCount = 3;

    activationFunction = sigmoid;
    activationDerivative = sigmoidDerivative;
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

    activationFunction = sigmoid;
    activationDerivative = sigmoidDerivative;
}

FFNeuralNetwork::FFNeuralNetwork(size_t inputNeurons, size_t inputBias, size_t numHidden, size_t* hiddenSizes, size_t* biasAmounts, size_t outputNeurons) {
    layerCount = numHidden + 2;

    layers = new NeuronLayer[numHidden + 2];

    layers[0].init(inputNeurons, inputBias);

    for(size_t i = 1; i < numHidden + 1; ++i) {
        layers[i].init(hiddenSizes[i], biasAmounts[i]);
    }

    layers[numHidden + 1].init(outputNeurons, 0);

    for(size_t i = 0; i < layerCount - 1; ++i) {
        layers[i].connectTo(layers[i+1]);
    }

    activationFunction = sigmoid;
    activationDerivative = sigmoidDerivative;
}

FFNeuralNetwork::~FFNeuralNetwork() {
    delete[] layers;
}

void FFNeuralNetwork::init(size_t inputNeurons, size_t inputBias, size_t numHidden, size_t neuronPerHidden, size_t biasPerHidden, size_t outputNeurons) {
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

    activationFunction = sigmoid;
    activationDerivative = sigmoidDerivative;
}

void FFNeuralNetwork::init(size_t inputNeurons, size_t inputBias, size_t numHidden, size_t* hiddenSizes, size_t* biasAmounts, size_t outputNeurons) {
    layerCount = numHidden + 2;

    layers = new NeuronLayer[numHidden + 2];

    layers[0].init(inputNeurons, inputBias);

    for(size_t i = 1; i < numHidden + 1; ++i) {
        layers[i].init(hiddenSizes[i], biasAmounts[i]);
    }

    layers[numHidden + 1].init(outputNeurons, 0);

    for(size_t i = 0; i < layerCount - 1; ++i) {
        layers[i].connectTo(layers[i+1]);
    }

    activationFunction = sigmoid;
    activationDerivative = sigmoidDerivative;
}

void FFNeuralNetwork::setFunctions(std::function<float(float)> activFunc, std::function<float(float)> deriv) {
    activationFunction = activFunc;
    activationDerivative = deriv;
}

void FFNeuralNetwork::setInputs(const float* vals, size_t arrSize) {
    if(arrSize == input().size()) {
        for(size_t i = 0; i < arrSize; ++i) {
            layers[0][i].setValue(vals[i]);
        }
    }
}

void FFNeuralNetwork::setInputs(const std::vector<float>& vals) {
    if(vals.size() == input().size()) {
        for(size_t i = 0; i < vals.size(); ++i) {
            layers[0][i].setValue(vals[i]);
        }
    }
}

void FFNeuralNetwork::propagateForwards() {
    input().sendWeightedVals();

    for(size_t i = 1; i < layerCount - 1; ++i) {
        layers[i].sendOutputs(activationFunction);
    }
}

std::vector<float> FFNeuralNetwork::getOutputs() {
    std::vector<float> outs;

    for(size_t i = 0; i < output().size(); ++i) {
        outs.push_back(layers[layerCount - 1][i].getOutput(activationFunction));
    }

    return outs;
}

std::vector<float> FFNeuralNetwork::processData(const float* vals, size_t arrSize) {
    setInputs(vals, arrSize);
    propagateForwards();
    std::vector<float> out = getOutputs();
    reset();
    return out;
}

std::vector<float> FFNeuralNetwork::processData(const std::vector<float>& vals) {
    setInputs(vals);
    propagateForwards();
    std::vector<float> out = getOutputs();
    reset();
    return out;
}

void FFNeuralNetwork::train(const float* inputData, size_t inputSize, const float* expected, size_t expectedSize, size_t numEpochs, float learningRate, float momentum) {

    srand(time(0));

    for(size_t i = 0; i < numEpochs; ++i) {

        size_t randIndex = rand() % (inputSize / input().size());

        setInputs(inputData+(randIndex*input().size()), input().size());

        propagateForwards();
        backPropagate(expected+(randIndex*output().size()), output().size(), learningRate, momentum);

        reset();
    }

}

void FFNeuralNetwork::train(const std::vector<float>& inputData, const std::vector<float>& expected, size_t numEpochs, float learningRate, float momentum) {

    srand(time(0));
    std::vector<float> tempInput;
    std::vector<float> tempExpected;

    tempInput.reserve(input().size());
    tempExpected.reserve(output().size());

    for(size_t i = 0; i < numEpochs; ++i) {
        size_t randIndex = rand() % (inputData.size() / input().size());

        for(size_t j = 0; j < input().size(); ++j) {
            tempInput[j] = inputData[randIndex + j];
        }

        for(size_t j = 0; j < output().size(); ++j) {
            tempExpected[j] = expected[randIndex + j];
        }

        setInputs(tempInput);

        propagateForwards();
        backPropagate(tempExpected, learningRate, momentum);

        reset();
    }
}

void FFNeuralNetwork::backPropagate(const float* expected, size_t numExpected, float learningRate, float momentum) {
    std::vector<float> previousErrors = calculateOutputError(expected, numExpected);
    adjustWeights(layers[layerCount - 2], previousErrors, learningRate, momentum);

    //start at the last hidden layer
    for(int i = layerCount - 2; i > 0; --i) {
        previousErrors = calculateHiddenError(layers[i], previousErrors);
        adjustWeights(layers[i-1], previousErrors, learningRate, momentum);
    }
}

void FFNeuralNetwork::backPropagate(const std::vector<float>& expected, float learningRate, float momentum) {
    std::vector<float> previousErrors = calculateOutputError(expected);
    adjustWeights(layers[layerCount - 2], previousErrors, learningRate, momentum);

    //start at the last hidden layer
    for(int i = layerCount - 2; i > 0; --i) {
        previousErrors = calculateHiddenError(layers[i], previousErrors);
        adjustWeights(layers[i-1], previousErrors, learningRate, momentum);
    }
}

std::vector<float> FFNeuralNetwork::calculateOutputError(const float* expected, size_t numExpected) {
    std::vector<float> errors;
    std::vector<float> outputs = getOutputs();

    for(size_t i = 0; i < output().size(); ++i) {
        errors.push_back(activationDerivative(layers[layerCount-1][i].getValue()) * (expected[i] - outputs[i]));
    }

    return errors;
}

std::vector<float> FFNeuralNetwork::calculateOutputError(const std::vector<float>& expected) {
    std::vector<float> errors;
    std::vector<float> outputs = getOutputs();

    for(size_t i = 0; i < output().size(); ++i) {
        errors.push_back(activationDerivative(layers[layerCount-1][i].getValue()) * (expected[i] - outputs[i]));
    }

    return errors;
}

std::vector<float> FFNeuralNetwork::calculateHiddenError(NeuronLayer& nl, std::vector<float> prevError) {
    std::vector<float> errors;

    for(size_t i = 0; i < nl.size(); ++i) {
        float deriv = activationDerivative(nl[i].getValue());

        float sumErr = 0.0f;

        if(!nl.getTarget()) {
            return errors;
        }

        for(size_t j = 0; j < nl.getTarget()->size(); ++j) {
            //weight * error
            sumErr += nl[i][j] * prevError[j];
        }

        errors.push_back(deriv * sumErr);
    }

    return errors;
}

void FFNeuralNetwork::adjustWeights(NeuronLayer& nl, std::vector<float> errors, float learningRate, float momentum) {
    if(!nl.getTarget()) {
        return;
    }

    for(size_t i = 0; i < nl.size(); ++i) {

        for(size_t j = 0; j < nl.getTarget()->size(); ++j) {

            float weightDelta = errors[j] * nl[i].getOutput(activationFunction) * learningRate;
            weightDelta += nl[i].getPreviousDeltas()[j] * momentum;
            nl[i].adjustWeight(j, weightDelta);

        }

    }
}

size_t FFNeuralNetwork::size() const {
    return layerCount;
}

void FFNeuralNetwork::reset() {
    for(size_t i = 0; i < layerCount; ++i) {
        layers[i].resetNeurons();
    }
}

NeuronLayer& FFNeuralNetwork::input() {
    if(layers) {
        return layers[0];
    } else {
        throw std::runtime_error("This neural network is uninitialized.");
    }
}

NeuronLayer& FFNeuralNetwork::output() {
    if(layers) {
        return layers[layerCount - 1];
    } else {
        throw std::runtime_error("This neural network is uninitialized.");
    }
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
    stream << "Network:\n";

    stream << "layerCount: " << ffnn.layerCount << "\n";
    stream << "inputSize: " << ffnn[0].size() << "\n";
    stream << "outputSize: " << ffnn[ffnn.size() - 1].size() << "\n";

    stream << "hiddenSizes: \n";

    for(size_t i = 1; i < ffnn.layerCount - 1; ++i) {
        stream << ffnn[i].size() << ",\n";
    }

    stream << "\n\n";

    for(size_t i = 0; i < ffnn.size(); ++i) {
        stream << ffnn[i] << "\n";
    }

    stream << "\n";

    return stream;
}

//------------------------------------------------------------------------//

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x) );
}

float sigmoidDerivative(float x) {
    float sig = sigmoid(x);
    return sig * (1.0f - sig);
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

NeuronLayer readLayer(std::string layer) {

}

Neuron readNeuron(std::string neuron) {
     std::stringstream valueReader(neuron);

     std::string line;

     Neuron n;

}
