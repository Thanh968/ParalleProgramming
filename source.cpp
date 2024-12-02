#include<iostream>
#include<fstream>
#include<string>
#include<vector>
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
        unsigned int required_mem_size = number_of_images * (n_rows * n_cols + 1);
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
                    inputData[i*n_rows*n_cols + r * n_cols + c + 1] = temp;
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

void forwardNN(float* input, float* weight, float* output, int inputRows, int inputCols, int outputCols, bool includeBias = true) {
    for (int i = 0; i < inputRows; i++) {
        for (int j = 0; j < outputCols; j++) {
            float temp = 0;

            for (int k = 0; k < inputCols; k++) {
                temp += input[i * inputCols + k] * weight[k * outputCols + j];
            }

            if (includeBias)
                output[i * outputCols + j + 1] = relu(temp);
            else {
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

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 10; j++) {
            cout << input_data[i*785 + j] << ' ';
        }
        cout << endl;
    }

    unsigned int requiredMemsizeForLabel = number_of_images * 10;
    input_labels = (float*)malloc(requiredMemsizeForLabel * sizeof(float));

    for (int i = 0; i < requiredMemsizeForLabel; i++) {
      input_labels[i] = 0;
    }

    read_labels(nameOfLabelFile, input_labels);

    // Cac ma tran trong so
    int inputLayerSize = 785;
    int firstHiddenLayerSize = 129;
    int secondHiddenLayerSize = 129;
    int lastHiddenLayerSize = 10;

    unsigned int sizeOfFirstWeight = inputLayerSize * (firstHiddenLayerSize - 1);
    unsigned int sizeOfSecondWeight = firstHiddenLayerSize * (secondHiddenLayerSize - 1);
    unsigned int sizeOfLastWeight = secondHiddenLayerSize * lastHiddenLayerSize;

    float *firstHiddenLayerWeight = NULL;
    float *secondHiddenLayerWeight = NULL;
    float *lastHiddenLayerWeight = NULL;

    firstHiddenLayerWeight = (float*)malloc(sizeOfFirstWeight * sizeof(float));
    secondHiddenLayerWeight = (float*)malloc(sizeOfSecondWeight * sizeof(float));
    lastHiddenLayerWeight = (float*)malloc(sizeOfLastWeight * sizeof(float));

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
    
    // Gan cot 0 cua cac ma tran ket qua voi gia tri mot
    for (int i = 0 ; i < number_of_images; i++) {
        firstLayerResult[i * firstHiddenLayerSize] = 1;
        secondLayerResult[i * secondHiddenLayerSize] = 1;
    }

    // Khoi tao trong so cho weight
    initialize_weights(firstHiddenLayerWeight,inputLayerSize,firstHiddenLayerSize - 1);
    initialize_weights(secondHiddenLayerWeight, firstHiddenLayerSize, secondHiddenLayerSize - 1);
    initialize_weights(lastHiddenLayerWeight, secondHiddenLayerSize, lastHiddenLayerSize);

    forwardNN(input_data, firstHiddenLayerWeight, firstLayerResult, number_of_images, inputLayerSize, firstHiddenLayerSize);
    forwardNN(firstLayerResult, secondHiddenLayerWeight, secondLayerResult, number_of_images, firstHiddenLayerSize, secondHiddenLayerSize);
    forwardNN(secondLayerResult, lastHiddenLayerWeight, lastLayerResult, number_of_images, secondHiddenLayerSize, lastHiddenLayerSize);

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < lastHiddenLayerSize; j++) {
            cout << lastLayerResult[i * lastHiddenLayerSize + j] << ' ';
        }
        cout << endl;
    }

    softmax(lastLayerResult, number_of_images, lastHiddenLayerSize);

    for (int i = 0; i < 5; i++) {
        float maxFloat = 0;
        int maxIndex;
        for (int j = 0; j < lastHiddenLayerSize; j++) {
            if (maxFloat < lastLayerResult[i*lastHiddenLayerSize + j]) {
                maxFloat = lastLayerResult[i*lastHiddenLayerSize + j];
                maxIndex = j;
            }
        }
        cout << "Image: " << i << endl;
        cout << "Label: " << maxIndex << endl;
    }

    free(input_data);
    free(input_labels);
    free(firstHiddenLayerWeight);
    free(secondHiddenLayerWeight);
    free(lastHiddenLayerWeight);
    free(firstLayerResult);
    free(secondLayerResult);
    free(lastLayerResult);
}