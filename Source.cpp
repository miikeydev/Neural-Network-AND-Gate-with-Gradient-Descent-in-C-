#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;  // Utilisation du namespace std pour �viter de r�p�ter 'std::'

// Fonction d'activation sigmo�de
float activationFunction(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Calcul de la sortie du neurone
float computeOutput(float x1, float x2, float w1, float w2, float bias) {
    return activationFunction(x1 * w1 + x2 * w2 + bias);
}

// Entra�nement du neurone avec descente de gradient
void trainNeuron(float data[][3], int epochs, float learningRate, ofstream& outputFile) {
    float w1 = 0.0, w2 = 0.0, bias = 0.0;
    float y, y_d, error, totalError;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        totalError = 0.0;

        for (int i = 0; i < 4; ++i) {
            float x1 = data[i][0];
            float x2 = data[i][1];
            y_d = data[i][2];

            y = computeOutput(x1, x2, w1, w2, bias);

            error = y_d - y;
            totalError += pow(error, 2);

            // Mise � jour des poids avec d�riv�e de la sigmo�de
            w1 += learningRate * error * y * (1 - y) * x1;
            w2 += learningRate * error * y * (1 - y) * x2;
            bias += learningRate * error * y * (1 - y);

            outputFile << "Epoch: " << epoch + 1 << ", y: " << y << ", y_d: " << y_d << ", CF: " << totalError
                << ", w1: " << w1 << ", w2: " << w2 << endl;
        }

        if (totalError == 0) break;  // Arr�ter si l'erreur totale est nulle
    }
}

int main() {
    // Donn�es de la table AND
    float data[4][3] = { {0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1} };
    int epochs = 100;
    float learningRate = 0.1;

    // Ouvrir un fichier pour �crire les r�sultats
    ofstream outputFile("output_results.txt");

    if (!outputFile.is_open()) {
        cout << "Erreur d'ouverture du fichier de sortie !" << endl;
        return -1;
    }

    // Entra�ner le neurone
    trainNeuron(data, epochs, learningRate, outputFile);

    outputFile.close();
    cout << "Entra�nement termin� et r�sultats enregistr�s dans output_results.txt\n";

    return 0;
}
