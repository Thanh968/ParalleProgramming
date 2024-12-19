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

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);                                                                 
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

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
    file.close();
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

void initialize_biases(double* bias, int n_in, int n_out) {
    for (int i = 0; i < n_out; i++) {
        bias[i] = double(rand()) / RAND_MAX * 2.0 - 1.0;
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
        gradient[c] = temp / rows;
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

void backwardNN(double* transposedMatrix, double* delta, double* gradient, int inHiddenLayerSize, int numSample, int outHiddenLayerSize) {
    for (int row = 0; row < inHiddenLayerSize; row++) {
        for (int col = 0; col < outHiddenLayerSize; col++) {
            double temp = 0;

            for (int index  = 0; index < numSample; index++) {
                temp += transposedMatrix[row * numSample + index] * delta[index * outHiddenLayerSize + col];
            }

            gradient[row * outHiddenLayerSize + col] = temp / numSample;
        }
    }
}

void trainNN(double* train_data, double* train_one_hot_labels, double* train_labels, double* val_data, double* val_labels, double* val_one_hot_labels, double* firstHiddenLayerWeight, double *secondHiddenLayerWeight, double *lastHiddenLayerWeight, double* firstBiases, double *secondBiases, double *lastBiases, int num_epoch, int numberOfTrainImage, int numberOfValImage, int inputCols, int firstHiddenLayerSize, int secondHiddenLayerSize, int lastHiddenLayerSize) {
    double* firstLayerResult, *secondLayerResult, *lastLayerResult;
    double* transposeSecondResult, *transposeFirstResult, *transposeInputMatrix;
    double* lastDelta, *secondDelta, *firstDelta;
    double* lastGradient, *secondGradient, *firstGradient;
    double* transposeLastWeight, *transposeSecondWeight;
    double* reluDerivativeSecondMatrix, *reluDerivativeFirstMatrix;
    double* thirdBiasGradient, *secondBiasGradient, *firstBiasGradient;

    transposeInputMatrix = (double*)malloc(numberOfTrainImage * inputCols * sizeof(double));
    firstLayerResult = (double*)malloc(numberOfTrainImage * firstHiddenLayerSize * sizeof(double));
    secondLayerResult = (double*)malloc(numberOfTrainImage * secondHiddenLayerSize * sizeof(double));
    lastLayerResult = (double*)malloc(numberOfTrainImage * lastHiddenLayerSize * sizeof(double));

    transposeSecondResult = (double*)malloc(numberOfTrainImage * secondHiddenLayerSize * sizeof(double));
    transposeFirstResult = (double*)malloc(numberOfTrainImage * firstHiddenLayerSize * sizeof(double));

    lastDelta = (double*)malloc(numberOfTrainImage * lastHiddenLayerSize * sizeof(double));
    secondDelta = (double*)malloc(numberOfTrainImage * secondHiddenLayerSize * sizeof(double));
    firstDelta = (double*)malloc(numberOfTrainImage * firstHiddenLayerSize * sizeof(double));

    lastGradient = (double*)malloc(secondHiddenLayerSize * lastHiddenLayerSize * sizeof(double));
    secondGradient = (double*)malloc(firstHiddenLayerSize * secondHiddenLayerSize * sizeof(double));
    firstGradient = (double*)malloc(inputCols * firstHiddenLayerSize * sizeof(double));

    transposeLastWeight = (double*)malloc(lastHiddenLayerSize * secondHiddenLayerSize * sizeof(double));
    transposeSecondWeight = (double*)malloc(secondHiddenLayerSize * firstHiddenLayerSize * sizeof(double));

    reluDerivativeSecondMatrix = (double*)malloc(numberOfTrainImage * secondHiddenLayerSize * sizeof(double));
    reluDerivativeFirstMatrix = (double*)malloc(numberOfTrainImage* firstHiddenLayerSize * sizeof(double));

    thirdBiasGradient = (double*)malloc(secondHiddenLayerSize  * lastHiddenLayerSize * sizeof(double));
    secondBiasGradient = (double*)malloc(firstHiddenLayerSize * secondHiddenLayerSize * sizeof(double));
    firstBiasGradient = (double*)malloc(inputCols * firstHiddenLayerSize * sizeof(double));

    transposeMatrix(train_data, transposeInputMatrix, numberOfTrainImage, inputCols);

    for (int i = 0; i < num_epoch; i++) {
        // Forward qua 3 lop
        forwardNN(train_data, firstHiddenLayerWeight, firstBiases, firstLayerResult,numberOfTrainImage, inputCols, firstHiddenLayerSize);
        forwardNN(firstLayerResult, secondHiddenLayerWeight, secondBiases, secondLayerResult, numberOfTrainImage, firstHiddenLayerSize, secondHiddenLayerSize);
        forwardNN(secondLayerResult, lastHiddenLayerWeight, lastBiases, lastLayerResult, numberOfTrainImage, secondHiddenLayerSize, lastHiddenLayerSize, false);
        // Goi ham softmax cho ket qua cua layer cuoi
        softmax(lastLayerResult, numberOfTrainImage, lastHiddenLayerSize);

        // backprop

        // Tinh transpose truoc
        transposeMatrix(secondLayerResult, transposeSecondResult, numberOfTrainImage, secondHiddenLayerSize);
        transposeMatrix(firstLayerResult, transposeFirstResult, numberOfTrainImage, firstHiddenLayerSize);

        calculateLastDelta(lastLayerResult, train_one_hot_labels, lastDelta, numberOfTrainImage, lastHiddenLayerSize);

        // Tinh cho gradient lop cuoi
        backwardNN(transposeSecondResult, lastDelta, lastGradient, secondHiddenLayerSize, numberOfTrainImage, lastHiddenLayerSize);
        gradientForBias(lastDelta, thirdBiasGradient, numberOfTrainImage, lastHiddenLayerSize);

        relu_derivative(secondLayerResult, reluDerivativeSecondMatrix, numberOfTrainImage, secondHiddenLayerSize);
        relu_derivative(firstLayerResult, reluDerivativeFirstMatrix, numberOfTrainImage, firstHiddenLayerSize);


        // Cho hidden layer 2

        //tinh delta
        transposeMatrix(lastHiddenLayerWeight, transposeLastWeight, secondHiddenLayerSize, lastHiddenLayerSize);
        multiplyMatrix(lastDelta, transposeLastWeight, secondDelta, numberOfTrainImage, lastHiddenLayerSize, secondHiddenLayerSize);
        multiplyMatrixElementWise(secondDelta, reluDerivativeSecondMatrix, secondDelta, numberOfTrainImage, secondHiddenLayerSize);

        //tinh gradient 
        // multiplyMatrix(transposedFirstResult, secondDelta, secondGradient, firstHiddenLayerSize, numTrainSamples, secondHiddenLayerSize);
        // devideMatrixToScalar(secondGradient, numTrainSamples, firstHiddenLayerSize, secondHiddenLayerSize);
        backwardNN(transposeFirstResult, secondDelta, secondGradient, firstHiddenLayerSize, numberOfTrainImage, secondHiddenLayerSize);
        gradientForBias(secondDelta, secondBiasGradient, numberOfTrainImage, secondHiddenLayerSize);
        

        // Cho hidden layer 1

        //tinh delta
        transposeMatrix(secondHiddenLayerWeight, transposeSecondWeight, firstHiddenLayerSize, secondHiddenLayerSize);
        multiplyMatrix(secondDelta, transposeSecondWeight, firstDelta, numberOfTrainImage, secondHiddenLayerSize, firstHiddenLayerSize);
        multiplyMatrixElementWise(firstDelta, reluDerivativeFirstMatrix, firstDelta, numberOfTrainImage, firstHiddenLayerSize);

        // tinh gradient 
        // tinh chuyen vi cua ma tran dau vao
        // multiplyMatrix(transposedInputMatrix, firstDelta, firstGradient, inputLayerSize, numTrainSamples, firstHiddenLayerSize);
        // devideMatrixToScalar(firstGradient, numTrainSamples, inputLayerSize, firstHiddenLayerSize);
        backwardNN(transposeInputMatrix, firstDelta, firstGradient, inputCols, numberOfTrainImage, firstHiddenLayerSize);
        gradientForBias(firstDelta, firstBiasGradient, numberOfTrainImage, firstHiddenLayerSize);
        
        // Cap nhat trong so
        // Layer 3
        updateWeights(lastHiddenLayerWeight, lastGradient, LEARNING_RATE, secondHiddenLayerSize, lastHiddenLayerSize);
        updateBias(lastBiases, thirdBiasGradient, LEARNING_RATE,lastHiddenLayerSize);

        // Layer 2
        updateWeights(secondHiddenLayerWeight, secondGradient, LEARNING_RATE, firstHiddenLayerSize, secondHiddenLayerSize);
        updateBias(secondBiases, secondBiasGradient, LEARNING_RATE,secondHiddenLayerSize);

        //layer 1
        updateWeights(firstHiddenLayerWeight, firstGradient, LEARNING_RATE, inputCols, firstHiddenLayerSize);
        updateBias(firstBiases, firstBiasGradient, LEARNING_RATE, firstHiddenLayerSize);

        forwardNN(train_data, firstHiddenLayerWeight, firstBiases, firstLayerResult, numberOfTrainImage, inputCols, firstHiddenLayerSize);
        forwardNN(firstLayerResult, secondHiddenLayerWeight, secondBiases, secondLayerResult, numberOfTrainImage, firstHiddenLayerSize, secondHiddenLayerSize);
        forwardNN(secondLayerResult, lastHiddenLayerWeight, lastBiases, lastLayerResult, numberOfTrainImage, secondHiddenLayerSize, lastHiddenLayerSize, false);

        softmax(lastLayerResult, numberOfTrainImage, lastHiddenLayerSize);

        cout << "Epoch: " << i ;
        cout <<", Train Accuracy: " << accuracy(lastLayerResult, train_labels, numberOfTrainImage, lastHiddenLayerSize) << \
            ", Train Loss: " << crossEntropy(lastLayerResult, train_one_hot_labels, numberOfTrainImage, lastHiddenLayerSize);

        forwardNN(val_data, firstHiddenLayerWeight, firstBiases, firstLayerResult, numberOfValImage, inputCols, firstHiddenLayerSize);
        forwardNN(firstLayerResult, secondHiddenLayerWeight, secondBiases, secondLayerResult, numberOfValImage, firstHiddenLayerSize, secondHiddenLayerSize);
        forwardNN(secondLayerResult, lastHiddenLayerWeight, lastBiases, lastLayerResult, numberOfValImage, secondHiddenLayerSize, lastHiddenLayerSize, false);

        softmax(lastLayerResult, numberOfValImage, lastHiddenLayerSize);

        cout <<", Val Accuracy: " << accuracy(lastLayerResult, val_labels, numberOfValImage, lastHiddenLayerSize) << \
            ", Val Loss: " << crossEntropy(lastLayerResult, val_one_hot_labels, numberOfValImage, lastHiddenLayerSize) << endl;
    }


    free(firstLayerResult);
    free(secondLayerResult);
    free(lastLayerResult);
    free(transposeFirstResult);
    free(transposeSecondResult);
    free(lastDelta);
    free(secondDelta);
    free(firstDelta);
    free(lastGradient);
    free(secondGradient);
    free(firstGradient);
    free(transposeLastWeight);
    free(transposeSecondWeight);
    free(reluDerivativeFirstMatrix);
    free(reluDerivativeSecondMatrix);
    free(transposeInputMatrix);
    free(thirdBiasGradient);
    free(secondBiasGradient);
    free(firstBiasGradient);
}

void readArrayFromFile(string filename, double* weight_1, double* weight_2, double* weight_3, double* bias_1, double* bias_2, double* bias_3, int inputLayerSize, int firstHiddenLayerSize, int secondHiddentLayerSize, int lastHiddenLayerSize) {
    // Mở file để đọc
    ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Can not open file\n";
        return;
    }

    // Đọc từng dòng và chuyển sang số thực
    double value;
    int count = 0;

    while (count < firstHiddenLayerSize * inputLayerSize) { 
        file >> value;
        weight_1[count] = value;
        count++;
    }

    count = 0;

    while (count < firstHiddenLayerSize) { 
        file >> value;
        bias_1[count] = value;
        count++;
    }

    count = 0; 

    while (count < firstHiddenLayerSize * secondHiddentLayerSize) { 
        file >> value;
        weight_2[count] = value;
        count++;
    }

    count = 0;

    while (count < secondHiddentLayerSize) { 
        file >> value;
        bias_2[count] = value;
        count++;
    }

    count = 0;

    while (count < secondHiddentLayerSize * lastHiddenLayerSize) { 
        file >> value;
        weight_3[count] = value;
        count++;
    }

    count = 0;

    while (count < lastHiddenLayerSize) { 
        file >> value;
        bias_3[count] = value;
        count++;
    }


    // Đóng file
    file.close();
    std::cout << "Complete reading file '" << filename << "'.\n";
}

int main() {
    srand(static_cast<unsigned>(time(0)));
    // Data file
    string filename = "data/train-images-idx3-ubyte";
    string nameOfLabelFile = "data/train-labels-idx1-ubyte";
    string valFileName = "data/val-images-idx3-ubyte";
    string valLabelFileName = "data/val-labels-idx1-ubyte";
    string weightFile = "weight.txt";

    // Input data
    double* input_data = NULL;
    double* input_one_hot_labels = NULL;
    double* input_labels = NULL;
    double* val_data = NULL;
    double* val_one_hot_labels = NULL;
    double* val_labels = NULL;

    // Doc Du lieu
    unsigned int number_of_train_images, n_train_rows, n_train_cols;
    unsigned int number_of_val_images, n_val_rows, n_val_cols; 
    read_mnist(filename, input_data, number_of_train_images, n_train_rows, n_train_cols);
    read_mnist(valFileName, val_data, number_of_val_images, n_val_rows, n_val_cols);

    unsigned int requiredMemsizeForTrainLabel = number_of_train_images * 10;
    unsigned int requiredMemsizeForValLabel = number_of_val_images * 10;
    input_one_hot_labels = (double*)malloc(requiredMemsizeForTrainLabel * sizeof(double));
    val_one_hot_labels = (double*)malloc(requiredMemsizeForValLabel * sizeof(double));

    for (int i = 0; i < number_of_train_images; i++) {
        for (int j = 0; j < 10; j++) {
            input_one_hot_labels[i * 10 + j] = 0;
        }
    }

    for (int i = 0; i < number_of_val_images; i++) {
        for (int j = 0; j < 10; j++) {
            val_one_hot_labels[i * 10 + j] = 0;
        }
    }

    read_labels_one_hot(nameOfLabelFile, input_one_hot_labels);
    read_labels_one_hot(valLabelFileName, val_one_hot_labels);

    input_labels = (double*)malloc(number_of_train_images * sizeof(double));
    val_labels = (double*)malloc(number_of_val_images * sizeof(double));

    read_labels(nameOfLabelFile, input_labels);
    read_labels(valLabelFileName, val_labels);

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

    // initialize_weights(firstHiddenLayerWeight,inputLayerSize,firstHiddenLayerSize);
    // initialize_weights(secondHiddenLayerWeight, firstHiddenLayerSize, secondHiddenLayerSize);
    // initialize_weights(lastHiddenLayerWeight, secondHiddenLayerSize, lastHiddenLayerSize);

    // Khoi tao bias
    double* firstBiases = NULL;
    double* secondBiases = NULL;
    double* lastBiases = NULL;

    firstBiases = (double*)malloc(firstHiddenLayerSize * sizeof(double));
    secondBiases = (double*)malloc(secondHiddenLayerSize * sizeof(double));
    lastBiases = (double*)malloc(lastHiddenLayerSize * sizeof(double));

    // initialize_biases(firstBiases, inputLayerSize, firstHiddenLayerSize);
    // initialize_biases(secondBiases, firstHiddenLayerSize, secondHiddenLayerSize);
    // initialize_biases(lastBiases, secondHiddenLayerSize, lastHiddenLayerSize);

    readArrayFromFile(weightFile, firstHiddenLayerWeight, secondHiddenLayerWeight, lastHiddenLayerWeight, firstBiases, secondBiases, lastBiases, inputLayerSize, firstHiddenLayerSize, secondHiddenLayerSize, lastHiddenLayerSize);

    for (int label = 0; label < 10; label++) {
        unsigned int count = 0;
        for (int image = 0; image < number_of_train_images; image++) {
            if (input_labels[image] == label) {
                count++;
            }
        }

        cout << "Label: " << label <<", Count: " << count << endl;
    }

    cout << "\nCheck label in val dataset\n\n";
    for (int label = 0; label < 10; label++) {
        unsigned int count = 0;
        for (int image = 0; image < number_of_val_images; image++) {
            if (val_labels[image] == label) {
                count++;
            }
        }

        cout << "Label: " << label << ", count: " << count << endl;
    }


    //##############################################################################################################################################
    trainNN(input_data, input_one_hot_labels, input_labels, val_data, val_labels, val_one_hot_labels, firstHiddenLayerWeight, secondHiddenLayerWeight, lastHiddenLayerWeight, firstBiases, secondBiases, lastBiases, NUM_EPOCH, number_of_train_images, number_of_val_images, inputLayerSize, firstHiddenLayerSize, secondHiddenLayerSize, lastHiddenLayerSize);
    //##############################################################################################################################################

    free(input_data);
    free(input_labels);
    free(input_one_hot_labels);
    free(firstHiddenLayerWeight);
    free(secondHiddenLayerWeight);
    free(lastHiddenLayerWeight);
    free(firstBiases);
    free(secondBiases);
    free(lastBiases);
    free(val_data);
    free(val_one_hot_labels);
    free(val_labels);
}