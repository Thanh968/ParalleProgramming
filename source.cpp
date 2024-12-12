#include<iostream>
#include<fstream>
#include<string>
#include<math.h>
#define LEARNING_RATE 0.1
#define NUM_EPOCH 10
#define TRAIN_RATE 0.8
#define VAL_RATE 0.1
#define TEST_RATE 0.1

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
                    inputData[i*n_rows*n_cols + r * n_cols + c] = double(temp) / 255;
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

void initialize_weights(double* weights, int n_in, int n_out)  {
    double stddev = sqrt(2.0 / n_in); 
    for (int i = 0; i < n_in * n_out; i++) {
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        double z = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2); 
        weights[i] = z * stddev;
    }
}

void initialize_biases(double* bias, int rows) {
    for (int i = 0; i < rows; i++) {
        bias[i] = ((rand() / (double)RAND_MAX) * 0.01) - 0.005;
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

double accuracy(double* lastLayerResult, double* labels, int rows, int cols) {
    double count = 0;
    for (int r = 0; r < rows; r++) {
        double maxProp = lastLayerResult[r * cols];
        int label = 0;

        for (int c = 0; c < cols; c++) {
            if (maxProp < lastLayerResult[r * cols + c]) {
                maxProp = lastLayerResult[r * cols + c];
                label = c;
            }
        }

        if (labels[r] == double(label)) {
            count++;
        }
    }
    double result = count / rows;
    return result;
}

// shuffle data

void shuffle_data(double* data, double* one_hot_label, double* labels, int rows, int cols, int one_hot_cols) {
    for (int row = rows - 1; row > 0; row--) {
        int swap_row_index = rand() % (row + 1);

        for (int col = 0; col < cols; col++) {
            double temp = data[row * cols + col];
            data[row * cols + col] = data[swap_row_index * cols + col];
            data[swap_row_index * cols + col] = temp; 
        }

        // swap one hot label

        for (int col = 0; col < one_hot_cols; col++) {
            double temp = one_hot_label[row * one_hot_cols + col];
            one_hot_label[row * one_hot_cols + col] = one_hot_label[swap_row_index * one_hot_cols + col];
            one_hot_label[swap_row_index * one_hot_cols + col] = temp;
        }

        // swap label

        double temp = labels[row];
        labels[row] = labels[swap_row_index];
        labels[swap_row_index] = temp;
    }
}

// phan chia du lieu

void split(double* data, double* one_hot_labels, double* labels, int rows, int cols, int one_hot_cols, \
           double* train_data, double* train_one_hot, double* train_labels,\
            double* val_data, double* val_one_hot, double* val_labels,\
            double* test_data, double* test_one_hot, double* test_labels) {
    int train_limit = TRAIN_RATE * rows;
    int val_limit = (TRAIN_RATE + VAL_RATE) * rows;
    int row = 0,train_index = 0, val_index = 0, test_index = 0;

    while (row < train_limit) {
        // copy data anh
        for (int col = 0; col < cols; col++) {
            train_data[train_index * cols + col] = data[row * cols + col];
        }

        // copy data one hot label
        for (int col = 0; col < one_hot_cols; col++) {
            train_one_hot[train_index * one_hot_cols + col] = one_hot_labels[row * one_hot_cols + col];
        }

        // copy label

        train_labels[train_index] = labels[row];
        row++;
        train_index++;
    }
    
    while (row < val_limit) {
        // copy data anh
        for (int col = 0; col < cols; col++) {
            val_data[val_index * cols + col] = data[row * cols + col];
        }

        // copy data one hot label
        for (int col = 0; col < one_hot_cols; col++) {
            val_one_hot[val_index * one_hot_cols + col] = one_hot_labels[row * one_hot_cols + col];
        }

        // copy label

        val_labels[val_index] = labels[row];
        row++;
        val_index++;
    }

    while (row < rows) {
        // copy data anh
        for (int col = 0; col < cols; col++) {
            test_data[test_index * cols + col] = data[row * cols + col];
        }

        // copy data one hot label
        for (int col = 0; col < one_hot_cols; col++) {
            test_one_hot[test_index * one_hot_cols + col] = one_hot_labels[row * one_hot_cols + col];
        }

        // copy label

        test_labels[test_index] = labels[row];
        row++;
        test_index++;
    }
}

void devideMatrixToScalar(double* matrix, double scalar, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] /= scalar;
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

    // Chia data
    double* train_data = NULL;
    double* train_one_hot_labels = NULL;
    double* train_labels = NULL;

    double* val_data = NULL;
    double* val_one_hot_labels = NULL;
    double* val_labels = NULL;

    double* test_data = NULL;
    double* test_one_hot_labels = NULL;
    double* test_labels = NULL;

    unsigned int sizeOfTrainData = TRAIN_RATE * number_of_images * inputLayerSize * sizeof(double);
    unsigned int sizeOfTrainOneHot = TRAIN_RATE * number_of_images * lastHiddenLayerSize * sizeof(double);
    unsigned int sizeOfTrainLabels = TRAIN_RATE * number_of_images * sizeof(double);

    unsigned int sizeOfValData = VAL_RATE * number_of_images * inputLayerSize * sizeof(double);
    unsigned int sizeOfValOneHot = VAL_RATE * number_of_images * lastHiddenLayerSize * sizeof(double);
    unsigned int sizeOfValLabels = VAL_RATE * number_of_images * sizeof(double);

    unsigned int sizeOfTestData = TEST_RATE * number_of_images * inputLayerSize * sizeof(double);
    unsigned int sizeOfTestOneHot = TEST_RATE * number_of_images * lastHiddenLayerSize * sizeof(double);
    unsigned int sizeOfTestLabels = TEST_RATE * number_of_images * sizeof(double);

    train_data = (double*)malloc(sizeOfTrainData);
    train_one_hot_labels = (double*)malloc(sizeOfTrainOneHot);
    train_labels = (double*)malloc(sizeOfTrainLabels);

    val_data = (double*)malloc(sizeOfValData);
    val_one_hot_labels = (double*)malloc(sizeOfValOneHot);
    val_labels = (double*)malloc(sizeOfValLabels);

    test_data = (double*)malloc(sizeOfTestData);
    test_one_hot_labels = (double*)malloc(sizeOfTestOneHot);
    test_labels = (double*)malloc(sizeOfTestLabels);

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

    transposeMatrix(input_data, transposedInputMatrix, number_of_images, inputLayerSize);


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

    shuffle_data(input_data, input_labels, labels, number_of_images, inputLayerSize, lastHiddenLayerSize);
    split(input_data, input_labels, labels, number_of_images, inputLayerSize, lastHiddenLayerSize, train_data, train_one_hot_labels, train_labels, val_data, val_one_hot_labels, val_labels, test_data, test_one_hot_labels, test_labels);

    cout << "Check label in train dataset\n\n";
    for (int label = 0; label < 10; label++) {
        unsigned int count = 0;
        for (int image = 0; image < TRAIN_RATE * number_of_images; image++) {
            if (train_labels[image] == label) {
                count++;
            }
        }

        cout << "Label: " << label <<", Count: " << count << endl;
    }
    int numTrainSamples = TRAIN_RATE * number_of_images;
    //===========================================================================================================================

    for (int i = 0; i < NUM_EPOCH; i++) {
        // Forward qua 3 lop

        forwardNN(train_data, firstHiddenLayerWeight, firstBiases, firstLayerResult, numTrainSamples, inputLayerSize, firstHiddenLayerSize);
        forwardNN(firstLayerResult, secondHiddenLayerWeight, secondBiases, secondLayerResult, numTrainSamples, firstHiddenLayerSize, secondHiddenLayerSize);
        forwardNN(secondLayerResult, lastHiddenLayerWeight, lastBiases, lastLayerResult, numTrainSamples, secondHiddenLayerSize, lastHiddenLayerSize, false);
        // Goi ham softmax cho ket qua cua layer cuoi
        softmax(lastLayerResult, numTrainSamples, lastHiddenLayerSize);
        // backprop

        // Tinh transpose truoc
        transposeMatrix(secondLayerResult, transposedSecondResult, numTrainSamples, secondHiddenLayerSize);
        transposeMatrix(firstLayerResult, transposedFirstResult, numTrainSamples, firstHiddenLayerSize);

        calculateLastDelta(lastLayerResult, train_one_hot_labels, lastDelta, numTrainSamples, lastHiddenLayerSize);

        // Tinh cho gradient lop cuoi
        multiplyMatrix(transposedSecondResult, lastDelta, lastGradient, secondHiddenLayerSize, numTrainSamples, lastHiddenLayerSize);
        devideMatrixToScalar(lastGradient, numTrainSamples, secondHiddenLayerSize, lastHiddenLayerSize);

        relu_derivative(secondLayerResult, reluDerivativeSecondMatrix, numTrainSamples, secondHiddenLayerSize);
        relu_derivative(firstLayerResult, reluDerivativeFirstMatrix, numTrainSamples, firstHiddenLayerSize);

        //gradientForBias(lastDelta, thirdBiasGradient, number_of_images, lastHiddenLayerSize);

        // Cho hidden layer 2

        //tinh delta
        transposeMatrix(lastHiddenLayerWeight, transposedLastWeight, secondHiddenLayerSize, lastHiddenLayerSize);
        multiplyMatrix(lastDelta, transposedLastWeight, secondDelta, numTrainSamples, lastHiddenLayerSize, secondHiddenLayerSize);
        multiplyMatrixElementWise(secondDelta, reluDerivativeSecondMatrix, secondDelta, numTrainSamples, secondHiddenLayerSize);

        //tinh gradient 
        multiplyMatrix(transposedFirstResult, secondDelta, secondGradient, firstHiddenLayerSize, numTrainSamples, secondHiddenLayerSize);
        devideMatrixToScalar(secondGradient, numTrainSamples, firstHiddenLayerSize, secondHiddenLayerSize);
        //gradientForBias(secondDelta, secondBiasGradient, number_of_images, secondHiddenLayerSize);

        // Cho hidden layer 1

        //tinh delta 
        transposeMatrix(secondHiddenLayerWeight, transposedSecondWeight, firstHiddenLayerSize, secondHiddenLayerSize);
        multiplyMatrix(secondDelta, transposedSecondWeight, firstDelta, numTrainSamples, secondHiddenLayerSize, firstHiddenLayerSize);
        multiplyMatrixElementWise(firstDelta, reluDerivativeFirstMatrix, firstDelta, numTrainSamples, firstHiddenLayerSize);

        // tinh gradient 
        // tinh chuyen vi cua ma tran dau vao
        multiplyMatrix(transposedInputMatrix, firstDelta, firstGradient, inputLayerSize, numTrainSamples, firstHiddenLayerSize);
        devideMatrixToScalar(firstGradient, numTrainSamples, inputLayerSize, firstHiddenLayerSize);
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
        cout << "Epoch: " << i ;
        cout <<", Train Accuracy: " << accuracy(lastLayerResult, train_labels, TRAIN_RATE * number_of_images, lastHiddenLayerSize) << \
            ", Train Loss: " << crossEntropy(lastLayerResult, train_one_hot_labels, TRAIN_RATE * number_of_images, lastHiddenLayerSize);

        forwardNN(val_data, firstHiddenLayerWeight, firstBiases, firstLayerResult, VAL_RATE * number_of_images, inputLayerSize, firstHiddenLayerSize);
        forwardNN(firstLayerResult, secondHiddenLayerWeight, secondBiases, secondLayerResult, VAL_RATE * number_of_images, firstHiddenLayerSize, secondHiddenLayerSize);
        forwardNN(secondLayerResult, lastHiddenLayerWeight, lastBiases, lastLayerResult, VAL_RATE * number_of_images, secondHiddenLayerSize, lastHiddenLayerSize, false);

        softmax(lastLayerResult, VAL_RATE * number_of_images, lastHiddenLayerSize);

        cout <<", Val Accuracy: " << accuracy(lastLayerResult, val_labels, VAL_RATE * number_of_images, lastHiddenLayerSize) << \
            ", Val Loss: " << crossEntropy(lastLayerResult, val_one_hot_labels, VAL_RATE * number_of_images, lastHiddenLayerSize) << endl;
    }

    //##############################################################################################################################################

    for (int i = 0; i < NUM_EPOCH; i++) {
        printf("%f ", errorList[i]);
    }
    cout << endl;
    cout << endl;
    forwardNN(test_data, firstHiddenLayerWeight, firstBiases, firstLayerResult, TEST_RATE * number_of_images, inputLayerSize, firstHiddenLayerSize);
    forwardNN(firstLayerResult, secondHiddenLayerWeight, secondBiases, secondLayerResult, TEST_RATE * number_of_images, firstHiddenLayerSize, secondHiddenLayerSize);
    forwardNN(secondLayerResult, lastHiddenLayerWeight, lastBiases, lastLayerResult, TEST_RATE * number_of_images, secondHiddenLayerSize, lastHiddenLayerSize, false);

    // Goi ham softmax cho ket qua cua layer cuoi
    softmax(lastLayerResult, TEST_RATE * number_of_images, lastHiddenLayerSize);
    cout << "Accuracy: " << accuracy(lastLayerResult, test_labels, TEST_RATE * number_of_images, lastHiddenLayerSize);

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
    free(train_data);
    free(train_one_hot_labels);
    free(train_labels);
    free(val_data);
    free(val_one_hot_labels);
    free(val_labels);
    free(test_data);
    free(test_one_hot_labels);
    free(test_labels);
}