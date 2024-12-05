#include<iostream>
#include<fstream>
#include<string>
#include<math.h>
#define LEARNING_RATE 0.3
#define NUM_EPOCH 10

using namespace std;

int reverseInt (unsigned int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((unsigned int)c1 << 24) + ((unsigned int)c2 << 16) + ((unsigned int)c3 << 8) + c4;
}

void read_mnist(string filename, double* &inputData, unsigned int& number_of_images, unsigned int& n_rows, unsigned int& n_cols)
{   
    ifstream file (filename);
    if (file.is_open())
    {
        unsigned int magic_number=0;
        unsigned char minNum = 255;
        unsigned char maxNum = 0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        cout << "Number of images: "<<number_of_images << endl;
        cout << "Number of rows: " << n_rows << endl;
        cout  << "Number of cols: " << n_cols << endl; 
        unsigned int required_mem_size = number_of_images * (n_rows * n_cols);
        inputData = (double*)malloc(required_mem_size * sizeof(double));
        for(int i=0;i<number_of_images;++i)
        {
            inputData[i*n_cols*n_rows] = 1;
            for(int r=0;r<n_rows;++r)
            {   
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    // Doc tung pixel
                    inputData[i*n_rows*n_cols + r * n_cols + c] = double(temp);
                }

            }
        }
    }
}

void read_labels_one_hot(string filename, double* inputLabel) {
    ifstream file (filename);
    if (file.is_open()) {
        unsigned int magic_number = 0;
        unsigned int number_of_label = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_label, sizeof(number_of_label));
        number_of_label = reverseInt(number_of_label);
        cout << "Magic number: " << magic_number << endl;
        cout << "Number of label: " << number_of_label << endl;

        for (int i = 0; i < number_of_label; i++) {
            // One hot cua cac label
            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            inputLabel[i*10 + int(temp)] = 1;
        }
    }
    file.close();
}

void read_labels(string filename, double* input_labels) {
    ifstream file (filename);
    if (file.is_open()) {
        unsigned int magic_number = 0;
        unsigned int number_of_label = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_label, sizeof(number_of_label));
        number_of_label = reverseInt(number_of_label);
        cout << "Magic number: " << magic_number << endl;
        cout << "Number of label: " << number_of_label << endl;

        for (int i = 0; i < number_of_label; i++) {
            // One hot cua cac label
            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            input_labels[i] = temp;
        }
    }
    file.close();
}

double relu(double x) {
    double result = (x > 0) ? x : 0;
    return result;
}

void forwardNN(double* input, double* weight, double* bias, double* output, int inputRows, int inputCols, int outputCols, bool usedActivate = true) {
    for (int i = 0; i < inputRows; i++) {
        for (int j = 0; j < outputCols; j++) {
            double temp = 0;
            for (int k = 0; k < inputCols; k++) {
                temp += input[i * inputCols + k] * weight[k * outputCols + j];
            }
            temp += bias[j];
            if (usedActivate) {
                output[i * outputCols + j] = relu(temp);
            } else {
                output[i * outputCols + j] = temp;
            }
        }
    }
}

void softmax(double* input, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;

        for (int j = 0; j < cols; j++) {
            double temp = exp(input[i * cols + j]);
            sum += temp;
            input[i*cols + j] = temp;
        }

        for (int j = 0; j < cols; j++) {
            input[i * cols + j] /= sum;
        }
    }
}

void initialize_weights(double* weights, int rows, int cols)  {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            weights[i * cols + j] = ((rand() / (double)RAND_MAX) * 0.001) - 0.0005;
        }
    }
}

void initialize_biases(double* bias, int rows) {
    for (int i = 0; i < rows; i++) {
        bias[i] = ((rand() / (double)RAND_MAX) * 0.001) - 0.0005;
    }
}

void calculateLastDelta(double* y_pred, double* y, double* delta, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            delta[i*cols + j] = y_pred[i*cols + j] - y[i*cols + j];
        }
    }
}

void multiplyMatrix(double* matrix_a, double* matrix_b, double* result,int rows_a, int cols_a, int cols_b) {
    for (int i = 0; i< rows_a; i++) {
        for (int j = 0; j < cols_b; j++) {
            double temp = 0.0;
            for (int k = 0; k < cols_a; k++) {
                temp += matrix_a[i*cols_a + k] * matrix_b[k * cols_b + j];
            }
            result[i*cols_b + j] = temp;
        }
    }
}

void transposeMatrix(double* inputMatrix, double* outputMatrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            outputMatrix[j * rows + i] = inputMatrix[i * cols + j];
        }
    }
}

void relu_derivative(double* input, double* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (input[i*cols + j] > 0) {
                output[i*cols + j] = 1;
            } else {
                output[i*cols + j] = 0;
            }
        }
    }
}

void multiplyMatrixElementWise(double* matrix_a, double* matrix_b, double* result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i * cols+ j] = matrix_a[i * cols + j] * matrix_b[i*cols + j];
        }
    }
}

void gradientForBias(double* delta, double* gradient,int rows, int cols) {
    for (int c = 0; c < cols; c++) {
        double temp = 0;
        for (int r = 0; r < rows; r++) {
            temp += delta[r*cols+c];
        }
        gradient[c] = temp;
    }
}

void updateWeights(double* weights, double* gradient, double lr, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            weights[i * cols +j] -= lr * gradient[i*cols + j];
        }
    }
}

void updateBias(double* bias, double* gradient, double lr, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        bias[i] -= lr * gradient[i];
    }
}

double crossEntropy(double* y_pred, double* groundTruthOneHot, int rows, int cols) {
    double result = 0;

    for (int i = 0; i < rows; i++) {
        double sumImage = 0;
        for (int j = 0; j < cols; j++) {
            sumImage += groundTruthOneHot[i * cols + j]*log(y_pred[i * cols + j]);
        }
        result += -sumImage;
    }

    result /= rows;
    return result;
}

// Nhan nghich dao ma tran a voi ma tran delta
// A_T * B
// rows_a = rows_b
void multiplyTransposeMaTrixA(double* matrix_a, double* matrix_b, double* result, int rows_a, int cols_a, int cols_b) {
    for (int i = 0; i < cols_a; i++) {
        for (int j = 0; j < cols_b; j++) {
            double temp = 0.0;

            for (int k = 0; k < rows_a; k++) {
                temp += matrix_a[k * cols_a + i] * matrix_b[k * cols_b + j];
            }

            result[i * cols_b + j] = temp;
        }
    }
}

int main() {
    srand(static_cast<unsigned>(time(0)));
    // Data file
    string filename = "train-images-idx3-ubyte";
    string nameOfLabelFile = "train-labels-idx1-ubyte";

    // Input data
    double* input_data = NULL;
    double* input_labels = NULL;

    // Doc Du lieu
    unsigned int number_of_images, n_rows, n_cols;
    read_mnist(filename, input_data, number_of_images, n_rows, n_cols);

    unsigned int requiredMemsizeForLabel = number_of_images * 10;
    input_labels = (double*)malloc(requiredMemsizeForLabel * sizeof(double));

    for (int i = 0; i < requiredMemsizeForLabel; i++) {
      input_labels[i] = 0;
    }

    read_labels_one_hot(nameOfLabelFile, input_labels);

    // Cac ma tran trong so

    int inputLayerSize = 784;
    int firstHiddenLayerSize = 128;
    int secondHiddenLayerSize = 128;
    int lastHiddenLayerSize = 10;

    // Khoi tao trong so
    unsigned int sizeOfFirstWeight = inputLayerSize * firstHiddenLayerSize;
    unsigned int sizeOfSecondWeight = firstHiddenLayerSize * secondHiddenLayerSize;
    unsigned int sizeOfLastWeight = secondHiddenLayerSize * lastHiddenLayerSize;

    double *firstHiddenLayerWeight = NULL;
    double *secondHiddenLayerWeight = NULL;
    double *lastHiddenLayerWeight = NULL;

    firstHiddenLayerWeight = (double*)malloc(sizeOfFirstWeight * sizeof(double));
    secondHiddenLayerWeight = (double*)malloc(sizeOfSecondWeight * sizeof(double));
    lastHiddenLayerWeight = (double*)malloc(sizeOfLastWeight * sizeof(double));

    initialize_weights(firstHiddenLayerWeight,inputLayerSize,firstHiddenLayerSize);
    initialize_weights(secondHiddenLayerWeight, firstHiddenLayerSize, secondHiddenLayerSize);
    initialize_weights(lastHiddenLayerWeight, secondHiddenLayerSize, lastHiddenLayerSize);

    // Khoi tao bias
    double* firstBiases = NULL;
    double* secondBiases = NULL;
    double* lastBiases = NULL;

    firstBiases = (double*)malloc(firstHiddenLayerSize * sizeof(double));
    secondBiases = (double*)malloc(secondHiddenLayerSize * sizeof(double));
    lastBiases = (double*)malloc(lastHiddenLayerSize * sizeof(double));

    initialize_biases(firstBiases, firstHiddenLayerSize);
    initialize_biases(secondBiases, secondHiddenLayerSize);
    initialize_biases(lastBiases, lastHiddenLayerSize);

    // Ma tran luu tru cac ket qua qua tung lop
    double* firstLayerResult = NULL;
    double* secondLayerResult = NULL;
    double* lastLayerResult = NULL;

    int sizeOfFirstLayerResult = number_of_images * firstHiddenLayerSize;
    int sizeOfSecondLayerResult = number_of_images * secondHiddenLayerSize;
    int sizeOfLastLayerResult = number_of_images * lastHiddenLayerSize;

    firstLayerResult = (double*)malloc(sizeOfFirstLayerResult * sizeof(double));
    secondLayerResult = (double*)malloc(sizeOfSecondLayerResult * sizeof(double));
    lastLayerResult = (double*)malloc(sizeOfLastLayerResult * sizeof(double));

    // Tao ma tran luu tru delta
    double* lastDelta = NULL;
    double* secondDelta = NULL;
    double* firstDelta = NULL;

    unsigned int lastDeltaSize = number_of_images * lastHiddenLayerSize;
    unsigned int secondDeltaSize = number_of_images * secondHiddenLayerSize;
    unsigned int firstDeltaSize = number_of_images * firstHiddenLayerSize;

    lastDelta = (double*)malloc(lastDeltaSize * sizeof(double));
    secondDelta = (double*)malloc(secondDeltaSize * sizeof(double));
    firstDelta = (double*)malloc(firstDeltaSize *sizeof(double));
    

    // Cap phat bo nho cho ma tran gradient
    double* lastGradient = NULL;
    double* secondGradient = NULL;
    double* firstGradient = NULL;

    lastGradient = (double*)malloc(sizeOfLastWeight * sizeof(double));
    secondGradient = (double*)malloc(sizeOfSecondWeight * sizeof(double));
    firstGradient = (double*)malloc(sizeOfFirstWeight * sizeof(double));

    // Cap phat bo nho cho ma tran chuyen vi
    
    double* transposedSecondResult = NULL;
    double* transposedFirstResult = NULL;
    double* transposedLastWeight = NULL;
    double* transposedSecondWeight = NULL;
    double* transposedInputMatrix = NULL;

    transposedSecondResult = (double*)malloc(sizeOfSecondLayerResult * sizeof(double));
    transposedFirstResult = (double*)malloc(sizeOfFirstLayerResult * sizeof(double));
    transposedLastWeight = (double*)malloc(sizeOfLastWeight * sizeof(double));
    transposedSecondWeight = (double*)malloc(sizeOfSecondWeight * sizeof(double));
    transposedInputMatrix = (double*)malloc((number_of_images * inputLayerSize) * sizeof(double));


    // Cap phat bo nho cho cac ma tran dao ham relu
    double* reluDerivativeSecondMatrix = NULL;
    double* reluDerivativeFirstMatrix = NULL;

    reluDerivativeSecondMatrix = (double*)malloc(sizeOfSecondLayerResult * sizeof(double));
    reluDerivativeFirstMatrix = (double*)malloc(sizeOfFirstLayerResult * sizeof(double));

    // Gradient cho bias
    double* firstBiasGradient = NULL;
    double* secondBiasGradient = NULL;
    double* thirdBiasGradient = NULL;

    firstBiasGradient = (double*)malloc(firstHiddenLayerSize * sizeof(double));
    secondBiasGradient = (double*)malloc(secondHiddenLayerSize * sizeof(double));
    thirdBiasGradient = (double*)malloc(lastHiddenLayerSize * sizeof(double));

    // Khoi tao mang chua do loi
    double* errorList = (double*) malloc(NUM_EPOCH * sizeof(double));

    for (int epoch = 0; epoch < NUM_EPOCH; epoch++) {
        errorList[epoch] = 0;
    }

    // Ground Truth Label
    double* labels = (double*)malloc(number_of_images*sizeof(double));
    read_labels(nameOfLabelFile, labels);

    for (int label = 0; label < 10; label++) {
        unsigned int count = 0;
        for (int image = 0; image < number_of_images; image++) {
            if (labels[image] == label) {
                count++;
            }
        }

        cout << "Label: " << label <<", Count: " << count << endl;
    }

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout << firstHiddenLayerWeight[i*firstHiddenLayerSize + j] << ' ';
        }
        cout << endl;
    }
    cout << endl;

    //===========================================================================================================================

    for (int i = 0; i < NUM_EPOCH; i++) {
        // Forward qua 3 lop

        forwardNN(input_data, firstHiddenLayerWeight, firstBiases, firstLayerResult, number_of_images, inputLayerSize, firstHiddenLayerSize);
        forwardNN(firstLayerResult, secondHiddenLayerWeight, secondBiases, secondLayerResult, number_of_images, firstHiddenLayerSize, secondHiddenLayerSize);
        forwardNN(secondLayerResult, lastHiddenLayerWeight, lastBiases, lastLayerResult, number_of_images, secondHiddenLayerSize, lastHiddenLayerSize, false);

        // Goi ham softmax cho ket qua cua layer cuoi
        softmax(lastLayerResult, number_of_images, lastHiddenLayerSize);
        for (int image = 0; image < 5; image++) {
            for(int col = 0; col < 10; col++) {
                cout << lastLayerResult[image * lastHiddenLayerSize + col] << ' ';
            }
            cout << endl;
        }

        // backprop

        // Tinh transpose truoc

        calculateLastDelta(lastLayerResult, input_labels,lastDelta, number_of_images, lastHiddenLayerSize);

        // Tinh cho gradient lop cuoi
        multiplyTransposeMaTrixA(secondLayerResult, lastDelta, lastGradient, number_of_images, secondHiddenLayerSize, lastHiddenLayerSize);

        relu_derivative(secondLayerResult, reluDerivativeSecondMatrix, number_of_images, secondHiddenLayerSize);
        relu_derivative(firstLayerResult, reluDerivativeFirstMatrix, number_of_images, firstHiddenLayerSize);

        //gradientForBias(lastDelta, thirdBiasGradient, number_of_images, lastHiddenLayerSize);

        // Cho hidden layer 2

        //tinh delta
        multiplyMatrix(lastDelta, transposedLastWeight, secondDelta, number_of_images, lastHiddenLayerSize, secondHiddenLayerSize);
        multiplyMatrixElementWise(secondDelta, reluDerivativeSecondMatrix, secondDelta, number_of_images, secondHiddenLayerSize);

        //tinh gradient 
        multiplyMatrix(transposedFirstResult, secondDelta, secondGradient, firstHiddenLayerSize, number_of_images, secondHiddenLayerSize);
        //gradientForBias(secondDelta, secondBiasGradient, number_of_images, secondHiddenLayerSize);

        // Cho hidden layer 1

        //tinh delta 
        multiplyMatrix(secondDelta, transposedSecondWeight, firstDelta, number_of_images, secondHiddenLayerSize, firstHiddenLayerSize);
        multiplyMatrixElementWise(firstDelta, reluDerivativeFirstMatrix, firstDelta, number_of_images, firstHiddenLayerSize);

        // tinh gradient 
        // tinh chuyen vi cua ma tran dau vao
        multiplyMatrix(transposedInputMatrix, firstDelta, firstGradient, inputLayerSize, number_of_images, firstHiddenLayerSize);
        //gradientForBias(firstDelta, firstBiasGradient, number_of_images, firstHiddenLayerSize);

        // Cap nhat trong so
        // Layer 3
        updateWeights(lastHiddenLayerWeight, lastGradient, LEARNING_RATE, secondHiddenLayerSize, lastHiddenLayerSize);
        //updateBias(lastBiases, thirdBiasGradient, LEARNING_RATE,lastHiddenLayerSize);

        // Layer 2
        updateWeights(secondHiddenLayerWeight, secondGradient, LEARNING_RATE, firstHiddenLayerSize, secondHiddenLayerSize);
        //updateBias(secondBiases, secondBiasGradient, LEARNING_RATE,secondHiddenLayerSize);

        //layer 1
        updateWeights(firstHiddenLayerWeight, firstGradient, LEARNING_RATE, inputLayerSize, firstHiddenLayerSize);
        //updateBias(firstBiases, firstBiasGradient, LEARNING_RATE, firstHiddenLayerSize);

        double err = crossEntropy(lastLayerResult, input_labels, number_of_images, lastHiddenLayerSize);
        errorList[i] = err;
    }

    //##############################################################################################################################################

    for (int i = 0; i < NUM_EPOCH; i++) {
        printf("%f ", errorList[i]);
    }
    cout << endl;
    cout << endl;

    free(transposedInputMatrix);
    free(input_data);
    free(input_labels);
    free(firstHiddenLayerWeight);
    free(secondHiddenLayerWeight);
    free(lastHiddenLayerWeight);
    free(firstLayerResult);
    free(secondLayerResult);
    free(lastLayerResult);
    free(lastGradient);
    free(secondGradient);
    free(firstGradient);
    free(lastDelta);
    free(secondDelta);
    free(firstDelta);
    free(transposedSecondResult);
    free(transposedFirstResult);
    free(transposedLastWeight);
    free(transposedSecondWeight);
    free(reluDerivativeFirstMatrix);
    free(reluDerivativeSecondMatrix);
    free(firstBiases);
    free(secondBiases);
    free(lastBiases);
    free(firstBiasGradient);
    free(secondBiasGradient);
    free(thirdBiasGradient);
    free(errorList);
    free(labels);
}