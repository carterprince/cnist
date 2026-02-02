#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#define INPUT_SIZE 784
#define HIDDEN1_SIZE 256
#define HIDDEN2_SIZE 256
#define OUTPUT_SIZE 10

int fix_integer(int n) {
    int a = n & 0xff000000;
    a = a >> 24;
    int b = n & 0x00ff0000;
    b = b >> 8;
    int c = n & 0x0000ff00;
    c = c << 8;
    int d = n & 0x000000ff;
    d = d << 24;

    return a + b + c + d;
}

double random_weight() {
    return ((random() / (double)RAND_MAX) - 0.5) * 0.25;
}

void main() {
    FILE *train_images_file = fopen("mnist/train-images-idx3-ubyte", "r");

    // skip magic
    fseek(train_images_file, sizeof(int), SEEK_CUR);

    int num_images;
    fread(&num_images, sizeof(int), 1, train_images_file);
    num_images = fix_integer(num_images);

    int rows;
    fread(&rows, sizeof(int), 1, train_images_file);
    rows = fix_integer(rows);

    int cols;
    fread(&cols, sizeof(int), 1, train_images_file);
    cols = fix_integer(cols);

    unsigned char ***train_images = malloc(num_images * sizeof(char*));
    for (int i = 0; i < num_images; i++) {
        unsigned char **image = malloc(rows * sizeof(char*));
        for (int j = 0; j < rows; j++) {
            unsigned char *row = malloc(cols * sizeof(char));
            fread(row, sizeof(char), cols, train_images_file);
            image[j] = row;
        }
        train_images[i] = image;
    }

    FILE *train_labels_file = fopen("mnist/train-labels-idx1-ubyte", "r");

    // skip magic and count
    fseek(train_labels_file, sizeof(int), SEEK_CUR);
    fseek(train_labels_file, sizeof(int), SEEK_CUR);

    unsigned char train_labels[num_images];
    fread(train_labels, sizeof(char), num_images, train_labels_file);

    FILE *test_images_file = fopen("mnist/t10k-images-idx3-ubyte", "r");

    // skip magic, rows, cols
    fseek(test_images_file, sizeof(int), SEEK_CUR);

    int num_test_images;
    fread(&num_test_images, sizeof(int), 1, test_images_file);
    num_test_images = fix_integer(num_test_images);

    fseek(test_images_file, sizeof(int), SEEK_CUR);
    fseek(test_images_file, sizeof(int), SEEK_CUR);

    unsigned char ***test_images = malloc(num_test_images * sizeof(char*));
    for (int i = 0; i < num_test_images; i++) {
        unsigned char **image = malloc(rows * sizeof(char*));
        for (int j = 0; j < rows; j++) {
            unsigned char *row = malloc(cols * sizeof(char));
            fread(row, sizeof(char), cols, test_images_file);
            image[j] = row;
        }
        test_images[i] = image;
    }


    FILE *test_labels_file = fopen("mnist/t10k-labels-idx1-ubyte", "r");

    // skip magic and count
    fseek(test_labels_file, sizeof(int), SEEK_CUR);
    fseek(test_labels_file, sizeof(int), SEEK_CUR);

    unsigned char test_labels[num_test_images];
    fread(test_labels, sizeof(char), num_test_images, test_labels_file);

    // 784 -> 16 -> 16 -> 10
    double **weights1 = malloc(HIDDEN1_SIZE * sizeof(double*));
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        double *neuron = malloc(INPUT_SIZE * sizeof(double));
        for (int j = 0; j < INPUT_SIZE; j++) {
            neuron[j] = random_weight();
        }
        weights1[i] = neuron;
    }
    double *biases1 = malloc(HIDDEN1_SIZE * sizeof(double));
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        biases1[i] = random_weight();
    }

    double **weights2 = malloc(HIDDEN2_SIZE * sizeof(double*));
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        double *neuron = malloc(HIDDEN1_SIZE * sizeof(double));
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            neuron[j] = random_weight();
        }
        weights2[i] = neuron;
    }
    double *biases2 = malloc(HIDDEN2_SIZE * sizeof(double));
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        biases2[i] = random_weight();
    }

    double **weights3 = malloc(OUTPUT_SIZE * sizeof(double*));
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double *neuron = malloc(HIDDEN2_SIZE * sizeof(double));
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            neuron[j] = random_weight();
        }
        weights3[i] = neuron;
    }
    double *biases3 = malloc(OUTPUT_SIZE * sizeof(double));
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        biases3[i] = random_weight();
    }

    // 784 -> 16 -> 16 -> 10
    double *activations1 = malloc(HIDDEN1_SIZE * sizeof(double));
    double *activations2 = malloc(HIDDEN2_SIZE * sizeof(double));
    double *activations3 = malloc(OUTPUT_SIZE * sizeof(double));

    double *errors1 = malloc(HIDDEN1_SIZE * sizeof(double));
    double *errors2 = malloc(HIDDEN2_SIZE * sizeof(double));
    double *errors3 = malloc(OUTPUT_SIZE * sizeof(double));

    double lr = 0.001;

    for (int TRAIN_IMAGE = 0; TRAIN_IMAGE < num_images; TRAIN_IMAGE++) {

        printf("training on example %d\n", TRAIN_IMAGE);

        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            activations1[i] = 0;
            for (int j = 0; j < INPUT_SIZE; j++) {
                int col = j % 28;
                int row = j / 28;
                activations1[i] += (train_images[TRAIN_IMAGE][row][col] / 255.0) * weights1[i][j];
            }
            activations1[i] += biases1[i];

            // ReLU
            if (activations1[i] < 0)
                activations1[i] = 0;
        }

        for (int i = 0; i < HIDDEN2_SIZE; i++) {
            activations2[i] = 0;
            for (int j = 0; j < HIDDEN1_SIZE; j++) {
                activations2[i] += activations1[j] * weights2[i][j];
            }
            activations2[i] += biases2[i];

            if (activations2[i] < 0)
                activations2[i] = 0;
        }

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            activations3[i] = 0;
            for (int j = 0; j < HIDDEN2_SIZE; j++) {
                activations3[i] += activations2[j] * weights3[i][j];
            }
            activations3[i] += biases3[i];

            if (activations3[i] < 0)
                activations3[i] = 0;

            if (activations3[i] > 1)
                activations3[i] = 1;
        }

        // calculate errors for each neuron
        // MSE loss
        double target[OUTPUT_SIZE];
        double mse = 0.0;
        double max_output = 0;
        int max_output_label = 0;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            target[i] = 0;
            if (i == train_labels[TRAIN_IMAGE]) {
                target[i] = 1.0;
            }
            if (activations3[i] > max_output) {
                max_output = activations3[i];
                max_output_label = i;
            }
            double error = 0;
            error = activations3[i] - target[i];
            errors3[i] = error;
            mse += errors3[i] * errors3[i];
            if (TRAIN_IMAGE % 100 == 0)
                printf("%d: %f\n", i, activations3[i]);
        }
        mse = mse / OUTPUT_SIZE;
        if (TRAIN_IMAGE % 100 == 0) {
            printf("true label: %d, ", train_labels[TRAIN_IMAGE]);
            printf("predicted label: %d\n", max_output_label);
            printf("mse: %f\n", mse);
        }

        for (int i = 0; i < HIDDEN2_SIZE; i++) {
            double error = 0;
            if (activations2[i] <= 0) {
                error = 0;
            } else {
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    error += errors3[j] * weights3[j][i];
                }
            }
            errors2[i] = error;
        }
        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            double error = 0;
            if (activations1[i] <= 0) {
                error = 0;
            } else {
                for (int j = 0; j < HIDDEN2_SIZE; j++) {
                    error += errors2[j] * weights2[j][i];
                }
            }
            errors1[i] = error;
        }

        // update weights
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN2_SIZE; j++) {
                double gradient = errors3[i] * activations2[j];
                weights3[i][j] = weights3[i][j] - (lr * gradient);
            }
            biases3[i] = biases3[i] - (lr * errors3[i]);
        }

        for (int i = 0; i < HIDDEN2_SIZE; i++) {
            for (int j = 0; j < HIDDEN1_SIZE; j++) {
                double gradient = errors2[i] * activations1[j];
                weights2[i][j] = weights2[i][j] - (lr * gradient);
            }
            biases2[i] = biases2[i] - (lr * errors2[i]);
        }

        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                int col = j % 28;
                int row = j / 28;
                double gradient = errors1[i] * (train_images[TRAIN_IMAGE][row][col] / 255.0);
                weights1[i][j] = weights1[i][j] - (lr * gradient);
            }
            biases1[i] = biases1[i] - (lr * errors1[i]);
        }

    }

    printf("testing model\n");

    // test
    for (int TEST_IMAGE = 0; TEST_IMAGE < num_test_images; TEST_IMAGE++) {
        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            activations1[i] = 0;
            for (int j = 0; j < INPUT_SIZE; j++) {
                int col = j % 28;
                int row = j / 28;
                activations1[i] += (test_images[TEST_IMAGE][row][col] / 255.0) * weights1[i][j];
            }
            activations1[i] += biases1[i];

            // ReLU
            if (activations1[i] < 0)
                activations1[i] = 0;
        }

        for (int i = 0; i < HIDDEN2_SIZE; i++) {
            activations2[i] = 0;
            for (int j = 0; j < HIDDEN1_SIZE; j++) {
                activations2[i] += activations1[j] * weights2[i][j];
            }
            activations2[i] += biases2[i];

            if (activations2[i] < 0)
                activations2[i] = 0;
        }

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            activations3[i] = 0;
            for (int j = 0; j < HIDDEN2_SIZE; j++) {
                activations3[i] += activations2[j] * weights3[i][j];
            }
            activations3[i] += biases3[i];

            if (activations3[i] < 0)
                activations3[i] = 0;

            if (activations3[i] > 1)
                activations3[i] = 1;
        }

        // calculate errors for each neuron
        // MSE loss
        double target[OUTPUT_SIZE];
        double mse = 0.0;
        double max_output = 0;
        int max_output_label = 0;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            target[i] = 0;
            if (i == test_labels[TEST_IMAGE]) {
                target[i] = 1.0;
            }
            if (activations3[i] > max_output) {
                max_output = activations3[i];
                max_output_label = i;
            }
            double error = 0;
            error = activations3[i] - target[i];
            errors3[i] = error;
            mse += errors3[i] * errors3[i];
            if (TEST_IMAGE % 100 == 0)
                printf("%d: %f\n", i, activations3[i]);
        }
        mse = mse / OUTPUT_SIZE;
        if (TEST_IMAGE % 100 == 0) {
            printf("true label: %d, ", test_labels[TEST_IMAGE]);
            printf("predicted label: %d\n", max_output_label);
            printf("mse: %f\n", mse);
        }
    }
}
