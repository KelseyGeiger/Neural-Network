#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <functional>
#include <utility>
#include <iostream>
#include <stdexcept>
#include <ctime>
#include <cstdlib>

class Neuron;

class Neuron {
    public:
        Neuron();

        Neuron(size_t w, float v = 0.0f, bool b = false);

        virtual ~Neuron();

        void init(size_t w);

        void setValue(float v);

        float getValue();

        void setWeight(size_t index, float value);

        float getWeight(size_t index);

        size_t weightCount();

        bool isBias();

        void setBias(bool b);

        float getOutput(std::function<float(float)> activationFunc);

        //make it easier to set up connections
        Neuron& getReference();

        const Neuron& getReference() const;

        void reset();

        void adjustWeight(size_t index, float delta);

        void adjustWeights(float* deltas, size_t numDeltas);

        const float* getPreviousDeltas() const;

        Neuron& operator+=(float toAdd);

        const float& operator[](size_t index) const;

        friend std::ostream& operator<<(std::ostream& stream, const Neuron& n);

    private:
        float* weights = nullptr;
        float* prevDeltas = nullptr;
        float value;

        bool bias;

        size_t numWeights;
};

#endif // NEURON_H
