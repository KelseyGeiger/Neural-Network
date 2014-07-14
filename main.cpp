#include <iostream>
#include "FFNeuralNetwork.h"

#include <chrono>
#include <cmath>

int main() {

    FFNeuralNetwork ffnnTest = FFNeuralNetwork(1, 1,
                                               2, 20, 1,
                                               1);

    std::ofstream testOutput;

    testOutput.open("testNet.nnw", std::ios_base::out | std::ios_base::trunc);

    std::vector<float> inputs, expected;

    for(size_t i = 0; i <= 359; ++i) {
        float rad = (float) i * (M_PI / 180.0f);

        inputs.push_back(rad);
        expected.push_back(std::sin(rad));
    }

    std::chrono::microseconds dur{0};

    std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();

    ffnnTest.train(inputs, expected, 1000000, 0.1f, 0.9f);

    dur = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);

    std::cout << "On average, training took " << (float) dur.count() / 1000000.0f << " microseconds.\n";

    float in[] = { 3.14159f };

    std::cout << "sine(3.14159): " << ffnnTest.processData(in, 1)[0] << "\n";
    std::cout << "Actually is: " << std::sin(in[0]) << "\n\n";

    in[0] = { M_PI / 2 };

    std::cout << "sine(pi / 2): " << ffnnTest.processData(in, 1)[0] << "\n";
    std::cout << "Actually is: " << std::sin(in[0]) << "\n\n";

    testOutput << ffnnTest << "\n";

    testOutput.close();

    return 0;
}
