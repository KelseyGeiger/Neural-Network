#include "Neuron.h"

Neuron::Neuron() {
    value = 0.0f;
    weights = nullptr;
    prevDeltas = nullptr;
    numWeights = 0;
    bias = false;
}

Neuron::Neuron(int w, float v, bool b) {
    numWeights = w;
    weights = new float[numWeights];
    prevDeltas = new float[numWeights];

    srand(time(0));

    for(size_t i = 0; i < numWeights; ++i) {
        weights[i] = ( (float) rand() / (float) RAND_MAX );
        prevDeltas[i] = 0.0f;
    }

    value = v;
    bias = b;
}

Neuron::~Neuron() {
    delete weights;
}

void Neuron::init(int w) {
    numWeights = w;
    weights = new float[numWeights];

    for(size_t i = 0; i < numWeights; ++i) {
        weights[i] = ( (float) rand() / (float) RAND_MAX );
        prevDeltas[i] = 0.0f;
    }
}

void Neuron::setValue(float v) {
    value = v;
}

float Neuron::getValue() {
    return value;
}

size_t Neuron::weightCount() {
    return numWeights;
}

bool Neuron::isBias() {
    return bias;
}

void Neuron::setBias(bool b) {
    bias = b;
}

float Neuron::getOutput(std::function<float(float)> activationFunc) {
    return activationFunc(value);
}

Neuron& Neuron::getReference() {
    return *this;
}

const Neuron& Neuron::getReference() const {
    return *this;
}

void Neuron::reset() {
    value = 0.0f;
}

void Neuron::adjustWeight(size_t index, float delta) {
    if(index < numWeights) {
        weights[index] += delta;
        prevDeltas[index] = delta;
    }
}

void Neuron::adjustWeights(float* deltas, size_t numDeltas) {
    if(numDeltas == numWeights) {
        for(size_t i = 0; i < numDeltas; ++i) {
            weights[i] += deltas[i];
            prevDeltas[i] = deltas[i];
        }
    }
}

const float* Neuron::getPreviousDeltas() const {
    return prevDeltas;
}

Neuron& Neuron::operator+=(float delta) {
    value += delta;
    return *this;
}

const float& Neuron::operator[](size_t index) const {
    if(index < numWeights - 1) {
        return weights[index];
    } else {
        throw std::out_of_range("There are fewer weights than the value of the given index.");
    }
}

std::ostream& operator<<(std::ostream& stream, const Neuron& n) {
    if(n.bias) {
        stream << "\t\tBias {\n";
    } else {
        stream << "\t\tNeuron {\n";
    }

    stream << "\t\t\tvalue = " << n.value << ";\n";
    stream << "\t\t\tweights = [\n";
    for(size_t i = 0; i < n.numWeights; ++i) {
        stream << "\t\t\t\t" << n[i];

        if(i < n.numWeights - 1) {
            stream << ",\n";
        } else {
            stream << "\n";
        }
    }
    stream << "\t\t\t];\n";
    stream << "\t\t};\n";

    return stream;
}
