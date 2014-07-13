#include "Neuron.h"

Neuron::Neuron() {
    value = 0.0f;
    weights = nullptr;
    prevDeltas = nullptr;
    numWeights = 0;
    bias = false;
}

Neuron::Neuron(size_t w, float v, bool b) {
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
    delete[] weights;
}

void Neuron::init(size_t w) {
    numWeights = w;

    delete[] weights;
    delete[] prevDeltas;

    weights = new float[numWeights];
    prevDeltas = new float[numWeights];

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

void Neuron::setWeight(size_t index, float value) {
    if(index < numWeights) {
        weights[index] = value;
    } else {
        throw std::out_of_range("There are fewer weights than " + std::to_string(index + 1));
    }
}

float Neuron::getWeight(size_t index) {
    if(index < numWeights) {
        return weights[index];
    } else {
        throw std::out_of_range("There are fewer weights than " + std::to_string(index + 1));
    }
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
    } else {
        throw std::out_of_range("There are fewer weights than " + std::to_string(index + 1));
    }
}

void Neuron::adjustWeights(float* deltas, size_t numDeltas) {
    if(numDeltas == numWeights) {
        for(size_t i = 0; i < numDeltas; ++i) {
            weights[i] += deltas[i];
            prevDeltas[i] = deltas[i];
        }
    } else {
        throw std::out_of_range("There are fewer weights than " + std::to_string(numDeltas));
    }
}
float Neuron::getPreviousDelta(size_t index) {
    if(index < numWeights) {
        return prevDeltas[index];
    } else {
        throw std::out_of_range("There are fewer weights than " + std::to_string(index + 1));
    }
}

const float* Neuron::getPreviousDeltas() const {
    return prevDeltas;
}

Neuron& Neuron::operator+=(float toAdd) {
    value += toAdd;
    return *this;
}

const float& Neuron::operator[](size_t index) const {
    if(index < numWeights) {
        return weights[index];
    } else {
        throw std::out_of_range("There are fewer weights than " + (index + 1));
    }
}

std::ostream& operator<<(std::ostream& stream, const Neuron& n) {
    stream << "bias: " << std::boolalpha << n.bias << "\n";
    stream << "value: " << n.value << "\n";
    stream << "weightCount: " << n.numWeights << "\n";
    stream << "weights:\n";
    for(size_t i = 0; i < n.numWeights; ++i) {
        stream << n[i] << ",\n";
    }

    stream << "\n";

    return stream;
}
