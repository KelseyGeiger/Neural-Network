#ifndef FFNEURALNETWORK_H
#define FFNEURALNETWORK_H

#include <math.h>
#include <fstream>
#include <sstream>

#include "NeuronLayer.h"

float logistic(float x);
float logisticDerivative(float x);
float hyperTan(float x);
float hyperTanDerivative(float x);

class FFNeuralNetwork;

FFNeuralNetwork loadNN(std::string filename);
NeuronLayer readLayer(std::string layer);
Neuron readNeuron(std::string neuron);

class FFNeuralNetwork
{
    public:
        FFNeuralNetwork();

        FFNeuralNetwork(const std::string& filename);

        FFNeuralNetwork(size_t inputNeurons, size_t inputBias, size_t numHidden, size_t neuronPerHidden, size_t biasPerHidden, size_t outputNeurons);
        FFNeuralNetwork(size_t inputNeurons, size_t inputBias, size_t numHidden, size_t* hiddenSizes, size_t* biasAmounts, size_t outputNeurons);

        virtual ~FFNeuralNetwork();

        void init(size_t inputNeurons, size_t inputBias, size_t numHidden, size_t neuronPerHidden, size_t biasPerHidden, size_t outputNeurons);
        void init(size_t inputNeurons, size_t inputBias, size_t numHidden, size_t* hiddenSizes, size_t* biasAmounts, size_t outputNeurons);

        void setFunctions(std::function<float(float)> activFunc = logistic, std::function<float(float)> deriv = logisticDerivative);

        void setInputs(const float* vals, size_t arrSize);
        void setInputs(const std::vector<float>& vals);

        void propagateForwards();

        std::vector<float> getOutputs();

        std::vector<float> processData(const float* vals, size_t arrSize);
        std::vector<float> processData(const std::vector<float>& vals);

        void train(const float* inputData, size_t inputSize, const float* expected, size_t expectedSize, size_t numEpochs = 2500, float learningRate = 1.0f, float momentum = 0.0f);
        void train(const std::vector<float>& inputData, const std::vector<float>& expected, size_t numEpochs = 2500, float learningRate = 1.0f, float momentum = 0.0f);

        void backPropagate(const float* expected, size_t numExpected, float learningRate, float momentum);
        void backPropagate(const std::vector<float>& expected, float learningRate, float momentum);

        std::vector<float> calculateOutputError(const float* expected, size_t numExpected);
        std::vector<float> calculateOutputError(const std::vector<float>& expected);

        std::vector<float> calculateHiddenError(NeuronLayer& nl, std::vector<float> prevError);

        void adjustWeights(NeuronLayer& nl, std::vector<float> errors, float learningRate, float momentum);

        size_t size() const;

        void reset();

        NeuronLayer& input();

        NeuronLayer& output();

        NeuronLayer& operator[](size_t index);

        const NeuronLayer& operator[](size_t index) const;

        friend std::ostream& operator<<(std::ostream& stream, const FFNeuralNetwork& ffnn);

    private:
        NeuronLayer* layers;

        size_t layerCount;

        std::function<float(float)> activationFunction;
        std::function<float(float)> activationDerivative;

};

#endif // FFNEURALNETWORK_H
