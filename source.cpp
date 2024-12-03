#include<iostream>
#include<fstream>
#include<string>
#include<math.h>

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

void read_mnist(string filename, float* &inputData, unsigned int& number_of_images, unsigned int& n_rows, unsigned int& n_cols)
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
        inputData = (float*)malloc(required_mem_size * sizeof(int));
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
                    inputData[i*n_rows*n_cols + r * n_cols + c] = temp;
                }

            }
        }
    }
}

void read_labels(string filename, float* inputLabel) {
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

float relu(float x) {
    float result = (x > 0) ? x : 0;
    return result;
}

void forwardNN(float* input, float* weight, float* bias, float* output, int inputRows, int inputCols, int outputCols, bool includeBias = true) {
    for (int i = 0; i < inputRows; i++) {
        for (int j = 0; j < outputCols; j++) {
            float temp = 0;

            for (int k = 0; k < inputCols; k++) {
                temp += input[i * inputCols + k] * weight[k * outputCols + j];
            }

            temp += bias[j];

            if (includeBias) {   
                output[i * outputCols + j] = relu(temp);
            } else {
                output[i * outputCols + j ] = relu(temp);
            }
        }
    }
}

void softmax(float* input, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0;

        for (int j = 0; j < cols; j++) {
            float temp = exp(input[i * cols + j]);
            sum += temp;
            input[i*cols + j] = temp;
        }

        for (int j = 0; j < cols; j++) {
            input[i * cols + j] /= sum;
        }
    }
}

void initialize_weights(float* weights, int rows, int cols)  {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            weights[i * cols + j] = ((float)rand())/RAND_MAX;
        }
    }
}

void initialize_biases(float* bias, int rows) {
    for (int i = 0; i < rows; i++) {
        bias[i] = float(rand()) / RAND_MAX;
    }
}

void calculateLastDelta(float* y_pred, float* y, float* delta, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            delta[i*cols + j] = y_pred[i*cols + j] - y[i*cols + j];
        }
    }
}

void multiplyMatrix(float* matrix_a, float* matrix_b, float* result,int rows_a, int cols_a, int cols_b) {
    for (int i = 0; i< rows_a; i++) {
        for (int j = 0; j < cols_b; j++) {
            float temp = 0.0;
            for (int k = 0; k < cols_a; k++) {
                temp += matrix_a[i*cols_a + k] * matrix_b[k * cols_b + j];
            }
            result[i*cols_b + j] = temp;
        }
    }
}

void transposeMatrix(float* inputMatrix, float* outputMatrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            outputMatrix[j * rows + i] = inputMatrix[i * cols + j];
        }
    }
}

void relu_derivative(float* input, float* output, int rows, int cols) {
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

void multiplyMatrixElementWise(float* matrix_a, float* matrix_b, float* result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i * cols+ j] = matrix_a[i * cols + j] * matrix_b[i*cols + j];
        }
    }
}

void gradientForBias(float* delta, float* gradient,int rows, int cols) {
    for (int c = 0; c < cols; c++) {
        float temp = 0;
        for (int r = 0; r < rows; r++) {
            temp += delta[r*cols+c];
        }
        gradient[c] = temp;
    }
}

int main() {

    // Data file
    string filename = "train-images-idx3-ubyte";
    string nameOfLabelFile = "train-labels-idx1-ubyte";

    // Input data
    float* input_data = NULL;
    float* input_labels = NULL;

    // Doc Du lieu
    unsigned int number_of_images, n_rows, n_cols;
    read_mnist(filename, input_data, number_of_images, n_rows, n_cols);

    unsigned int requiredMemsizeForLabel = number_of_images * 10;
    input_labels = (float*)malloc(requiredMemsizeForLabel * sizeof(float));

    for (int i = 0; i < requiredMemsizeForLabel; i++) {
      input_labels[i] = 0;
    }

    read_labels(nameOfLabelFile, input_labels);

    // Cac ma tran trong so
    int inputLayerSize = 784;
    int firstHiddenLayerSize = 128;
    int secondHiddenLayerSize = 128;
    int lastHiddenLayerSize = 10;

    // Them 1 thay cho bias
    unsigned int sizeOfFirstWeight = inputLayerSize * firstHiddenLayerSize;
    unsigned int sizeOfSecondWeight = firstHiddenLayerSize * secondHiddenLayerSize;
    unsigned int sizeOfLastWeight = secondHiddenLayerSize * lastHiddenLayerSize;

    float *firstHiddenLayerWeight = NULL;
    float *secondHiddenLayerWeight = NULL;
    float *lastHiddenLayerWeight = NULL;

    firstHiddenLayerWeight = (float*)malloc(sizeOfFirstWeight * sizeof(float));
    secondHiddenLayerWeight = (float*)malloc(sizeOfSecondWeight * sizeof(float));
    lastHiddenLayerWeight = (float*)malloc(sizeOfLastWeight * sizeof(float));

    // Khoi tao bias
    float* firstBiases = NULL;
    float* secondBiases = NULL;
    float* lastBiases = NULL;

    firstBiases = (float*)malloc(firstHiddenLayerSize * sizeof(float));
    secondBiases = (float*)malloc(secondHiddenLayerSize * sizeof(float));
    lastBiases = (float*)malloc(lastHiddenLayerSize * sizeof(float));

    initialize_biases(firstBiases, firstHiddenLayerSize);
    initialize_biases(secondBiases, secondHiddenLayerSize);
    initialize_biases(lastBiases, lastHiddenLayerSize);

    // Ma tran luu tru cac ket qua qua tung lop
    float* firstLayerResult = NULL;
    float* secondLayerResult = NULL;
    float* lastLayerResult = NULL;

    int sizeOfFirstLayerResult = number_of_images * firstHiddenLayerSize;
    int sizeOfSecondLayerResult = number_of_images * secondHiddenLayerSize;
    int sizeOfLastLayerResult = number_of_images * lastHiddenLayerSize;

    firstLayerResult = (float*)malloc(sizeOfFirstLayerResult * sizeof(float));
    secondLayerResult = (float*)malloc(sizeOfSecondLayerResult * sizeof(float));
    lastLayerResult = (float*)malloc(sizeOfLastLayerResult * sizeof(float));
    

    // Khoi tao trong so cho weight
    initialize_weights(firstHiddenLayerWeight,inputLayerSize,firstHiddenLayerSize);
    initialize_weights(secondHiddenLayerWeight, firstHiddenLayerSize, secondHiddenLayerSize);
    initialize_weights(lastHiddenLayerWeight, secondHiddenLayerSize, lastHiddenLayerSize);

    forwardNN(input_data, firstHiddenLayerWeight, firstBiases, firstLayerResult, number_of_images, inputLayerSize, firstHiddenLayerSize);
    forwardNN(firstLayerResult, secondHiddenLayerWeight, secondBiases, secondLayerResult, number_of_images, firstHiddenLayerSize, secondHiddenLayerSize);
    forwardNN(secondLayerResult, lastHiddenLayerWeight, lastBiases, lastLayerResult, number_of_images, secondHiddenLayerSize, lastHiddenLayerSize);

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < lastHiddenLayerSize; j++) {
            cout << lastLayerResult[i * lastHiddenLayerSize + j] << ' ';
        }
        cout << endl;
    }

    softmax(lastLayerResult, number_of_images, lastHiddenLayerSize);


}