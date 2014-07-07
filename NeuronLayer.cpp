#include "NeuronLayer.h"

NeuronLayer::NeuronLayer() {
    neuronCount = 0;
    biasCount = 0;

    neurons = nullptr;
    target = nullptr;
}

NeuronLayer::NeuronLayer(size_t numNeurons, size_t numBias) {
    neuronCount = numNeurons;
    biasCount = numBias;

    target = nullptr;

    neurons = new Neuron[neuronCount + biasCount];

    for(size_t i = 0; i < neuronCount; ++i) {
        neurons[i].setValue(0.0f);
    }

    srand(time(0));

    for(size_t i = neuronCount; i < (neuronCount + biasCount); ++i) {
        neurons[i].setBias(true);
        neurons[i].setValue((float) rand() / (float) RAND_MAX);
    }
}

NeuronLayer::~NeuronLayer() {
    delete[] neurons;
}

void NeuronLayer::init(size_t numNeurons, size_t numBias) {
    neuronCount = numNeurons;
    biasCount = numBias;

    target = nullptr;

    neurons = new Neuron[neuronCount + biasCount];

    for(size_t i = 0; i < neuronCount; ++i) {
        neurons[i].setValue(0.0f);
    }

    srand(time(0));

    for(size_t i = neuronCount; i < (neuronCount + biasCount); ++i) {
        neurons[i].setBias(true);
        neurons[i].setValue((float) rand() / (float) RAND_MAX);
    }
}

void NeuronLayer::connectTo(NeuronLayer& nextLayer) {
    target = &nextLayer;

    srand(time(0));

    for(size_t i = 0; i < (neuronCount + biasCount); ++i) {
        neurons[i].init(nextLayer.size());

        for(size_t j = 0; j < nextLayer.size(); ++j) {
            neurons[i].setWeight(j, ((float) rand() / (float) RAND_MAX) - 0.5f);
        }
    }
}

void NeuronLayer::resetNeurons() {
    for(size_t i = 0; i < neuronCount; ++i) {
        neurons[i].reset();
    }
}

void NeuronLayer::sendWeightedVals() {
    for(size_t i = 0; i < (neuronCount + biasCount); ++i) {
        for(size_t j = 0; j < target->size(); ++j) {
            (*target)[j] += ( neurons[i][j] * neurons[i].getValue() );
        }
    }
}

void NeuronLayer::sendOutputs(std::function<float(float)> activFunc) {
    for(size_t i = 0; i < (neuronCount + biasCount); ++i) {
        for(size_t j = 0; j < target->size(); ++j) {
            (*target)[j] += ( neurons[i][j] * neurons[i].getOutput(activFunc));
        }
    }
}

size_t NeuronLayer::size() {
    return neuronCount;
}

Neuron& NeuronLayer::operator[](size_t index) {
    if(index < neuronCount + biasCount) {
        return neurons[index];
    } else {
        throw std::out_of_range("There are fewer neurons than the value of the given index + 1.");
    }
}

const Neuron& NeuronLayer::operator[](size_t index) const {
    if(index < neuronCount + biasCount) {
        return neurons[index];
    } else {
        throw std::out_of_range("There are fewer neurons than the value of the given index + 1.");
    }
}

std::ostream& operator<<(std::ostream& stream, const NeuronLayer& nl) {
    stream << "\tLayer {\n";
    stream << "\t\tneuronCount = " << nl.neuronCount << ";\n";
    stream << "\t\tbiasCount = " << nl.biasCount << ";\n\n";

    for(size_t i = 0; i < (nl.neuronCount + nl.biasCount); ++i) {
        stream << nl[i] << "\n\n";
    }

    stream << "\t};\n";

    return stream;
}
