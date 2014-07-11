#include <iostream>
#include "FFNeuralNetwork.h"

int main() {

    FFNeuralNetwork ffnnTest = FFNeuralNetwork(2, 1, 1, 2, 1, 1);

    std::ofstream testOutput;

    testOutput.open("testNet.nnw", std::ios_base::out | std::ios_base::trunc);

    testOutput << ffnnTest << "\n";

    testOutput.close();

    return 0;
}
