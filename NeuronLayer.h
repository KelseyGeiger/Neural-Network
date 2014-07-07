#ifndef NEURONLAYER_H
#define NEURONLAYER_H

#include "Neuron.h"

class NeuronLayer {
    public:
        NeuronLayer();

        NeuronLayer(size_t numNeurons, size_t numBias);

        virtual ~NeuronLayer();

        void init(size_t numNeurons, size_t numBias);

        void connectTo(NeuronLayer& nextLayer);

        void resetNeurons();

        void sendWeightedVals();

        void sendOutputs(std::function<float(float)> activFunc);

        size_t size();

        Neuron& operator[](size_t index);

        const Neuron& operator[](size_t index) const;

        friend std::ostream& operator<<(std::ostream& stream, const NeuronLayer& nl);

    private:
        Neuron* neurons = nullptr;

        NeuronLayer* target = nullptr;

        size_t neuronCount;
        size_t biasCount;
};

#endif // NEURONLAYER_H
