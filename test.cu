#include<iostream>
#include<fstream>
#include<string>
#include<math.h>
#define LEARNING_RATE 0.1
#define NUM_EPOCH 1
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
        inputData = (double*)malloc(required_mem_size * sizeof(int));
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

void read_labels(string filename, double* inputLabel) {
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
            weights[i * cols + j] = ((double)rand())/RAND_MAX;
        }
    }
}

void initialize_biases(double* bias, int rows) {
    for (int i = 0; i < rows; i++) {
        bias[i] = double(rand()) / RAND_MAX;
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

void printMatrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << matrix[i * cols + j] << ' ';
        }
        cout << endl;
    }
}

// Nhan A_T * B
void multiplyTransposeAndMatrix(double* matrix_A,  double* matrix_b, double* result, unsigned int rows_a, unsigned int cols_a, unsigned int cols_b) {
    for (int col_a = 0; col_a < cols_a; col_a++) {
        for (int col_b = 0; col_b < cols_b; col_b++) {
            double temp = 0;

            for (int index = 0; index < rows_a; index++) {
                temp += matrix_A[index * cols_a + col_a] * matrix_b[index * cols_b + col_b];
            }

            result[col_a * cols_b + col_b] = temp;
        }
    }
}


// Nhan A * B_T
// cols_a = cols_b
void multiplyMatrixAndTranspose(double* matrix_A, double* matrix_b, double* result, int rows_a, int cols_a, int rows_b) {
    for (int row_a = 0; row_a < rows_a; row_a++) {
        for (int row_b = 0; row_b < rows_b; row_b++) {
            double temp = 0;

            for (int index = 0; index < cols_a; index++) {
                temp += matrix_A[row_a * cols_a + index] * matrix_b[row_b * cols_a + index];
            }

            result[row_a * rows_b + row_b] = temp;
        }
    }
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
    int train_limit = 0.8 * rows;
    int val_limit = 0.9 * rows;
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

void initMatrix(double* matrix, int rows, int cols) {
    for (int row = 0; row <  rows; row++) {
        for (int col = 0; col < cols; col++) {
            int index = row * cols + col;
            matrix[index] = (rand() / RAND_MAX * 10);
        }
    }
}

int main() {
    srand(static_cast<unsigned>(time(0)));
    unsigned int rows = 1000;
    unsigned int cols = 1000;
    unsigned int rows_weight = 1000;
    unsigned int cols_weight = 500;
    double* data, *weight, *result, *bias;

    data = (double*)malloc(rows * cols * sizeof(double));
    weight = (double*)malloc(rows_weight * cols_weight * sizeof(double));
    result = (double*)malloc(rows * cols_weight * sizeof(double));
    bias = (double*)malloc(cols_weight * sizeof(double));

    GpuTimer timer;
    timer.Start();
    forwardNN(data, weight, bias, result, rows, cols, cols_weight);
    timer.Stop();
    cout << "Thoi gian can cho forward" << endl;

    free(data);
    free(weight);
    free(result);
    free(bias);
}