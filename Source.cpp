#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;  // Utilisation du namespace std pour éviter de répéter 'std::'

// Fonction d'activation sigmoïde
float activationFunction(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Calcul de la sortie du neurone
float computeOutput(float x1, float x2, float w1, float w2, float bias) {
    return activationFunction(x1 * w1 + x2 * w2 + bias);
}

// Entraînement du neurone avec descente de gradient
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

            // Mise à jour des poids avec dérivée de la sigmoïde
            w1 += learningRate * error * y * (1 - y) * x1;
            w2 += learningRate * error * y * (1 - y) * x2;
            bias += learningRate * error * y * (1 - y);

            outputFile << "Epoch: " << epoch + 1 << ", y: " << y << ", y_d: " << y_d << ", CF: " << totalError
                << ", w1: " << w1 << ", w2: " << w2 << endl;
        }

        if (totalError == 0) break;  // Arrêter si l'erreur totale est nulle
    }
}

int main() {
    // Données de la table AND
    float data[4][3] = { {0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1} };
    int epochs = 100;
    float learningRate = 0.1;

    // Ouvrir un fichier pour écrire les résultats
    ofstream outputFile("output_results.txt");

    if (!outputFile.is_open()) {
        cout << "Erreur d'ouverture du fichier de sortie !" << endl;
        return -1;
    }

    // Entraîner le neurone
    trainNeuron(data, epochs, learningRate, outputFile);

    outputFile.close();
    cout << "Entraînement terminé et résultats enregistrés dans output_results.txt\n";

    return 0;
}
