#include <iostream>
#include "FFNeuralNetwork.h"

#include <chrono>

int main() {

    FFNeuralNetwork ffnnTest = FFNeuralNetwork(2, 1,
                                               1, 3, 1,
                                               1);

    std::ofstream testOutput;

    testOutput.open("testNet.nnw", std::ios_base::out | std::ios_base::trunc);

    float inputs[] = {
        1, 1,
        1, 0,
        0, 0,
        0, 1
    };

    float expected[] = {
        1,
        0,
        0,
        0
    };

    std::chrono::microseconds dur{0};

    std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();

    ffnnTest.train(inputs, 8, expected, 4, 50000, 0.02f, 0.0f);

    dur = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);

    std::cout << "On average, training took " << (float) dur.count() / 50000.0f << " microseconds.\n";

    float in[] = {
        1, 1
    };

    std::cout << "1 AND 1 : " << ffnnTest.processData(inputs, 2)[0] << "\n";

    in[0] = 1;
    in[1] = 0;

    std::cout << "1 AND 0 : " << ffnnTest.processData(inputs, 2)[0] << "\n";

    in[0] = 0;
    in[1] = 1;

    std::cout << "0 AND 1 : " << ffnnTest.processData(inputs, 2)[0] << "\n";

    in[0] = 0;
    in[1] = 0;

    std::cout << "0 AND 0 : " << ffnnTest.processData(inputs, 2)[0] << "\n";

    testOutput << ffnnTest << "\n";

    testOutput.close();

    return 0;
}
