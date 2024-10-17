#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>   

using namespace std;

// Sigmoid activation function
float activationFunction(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// Compute the neuron's output
float computeOutput(float x1, float x2, float w1, float w2, float bias) {
    return activationFunction(x1 * w1 + x2 * w2 + bias);
}

// Train the neuron with gradient descent
void trainNeuron(float data[][3], int epochs, float learningRate, ofstream& outputFile) {
    // Initialize weights and bias to small random values
    float w1 = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    float w2 = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    float bias = static_cast<float>(rand()) / RAND_MAX - 0.5f;

    float y, y_d, error, totalError;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        totalError = 0.0f;

        for (int i = 0; i < 4; ++i) {
            float x1 = data[i][0];
            float x2 = data[i][1];
            y_d = data[i][2];

            y = computeOutput(x1, x2, w1, w2, bias);

            error = y_d - y;
            totalError += pow(error, 2);

            // Update weights and bias with sigmoid derivative
            float delta = learningRate * error * y * (1 - y);
            w1 += delta * x1;
            w2 += delta * x2;
            bias += delta;

            outputFile << "Epoch: " << epoch + 1 << ", y: " << y << ", y_d: " << y_d
                << ", CF: " << totalError << ", w1: " << w1 << ", w2: " << w2 << endl;
        }

        if (totalError < 0.001f) break; // Stop if total error is small enough
    }

    // Output the final weights and bias
    outputFile << "Final weights and bias: w1 = " << w1 << ", w2 = " << w2 << ", bias = " << bias << endl;
}

int main() {
    // Initialize random seed
    srand(static_cast<unsigned int>(time(0)));

    // AND truth table data
    float data[4][3] = { {0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1} };
    int epochs = 10000;
    float learningRate = 0.1f;

    // Open a file to write results
    ofstream outputFile("output_results.txt");

    if (!outputFile.is_open()) {
        cout << "Error opening output file!" << endl;
        return -1;
    }

    // Train the neuron
    trainNeuron(data, epochs, learningRate, outputFile);

    outputFile.close();
    cout << "Training completed. Results saved to output_results.txt\n";

    return 0;
}
