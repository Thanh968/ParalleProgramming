#include<iostream>
#include<fstream>
#include<string>
#include<vector>

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

void read_mnist(string filename, unsigned int* inputData, unsigned int& number_of_images, unsigned int& n_rows, unsigned int& n_cols)
{   
    ifstream file (filename);
    if (file.is_open())
    {
        unsigned int magic_number=0;
        // unsigned int number_of_images=0;
        // unsigned int n_rows=0;
        // unsigned int n_cols=0;
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
        unsigned int required_mem_size = number_of_images * n_rows * n_cols;
        inputData = (unsigned int*)malloc(required_mem_size * sizeof(unsigned int));
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    inputData[i*n_rows*n_cols + r * n_cols + c] = temp;
                    if (minNum > temp) {
                        minNum = temp;
                    }
                    
                    if (maxNum < temp) {
                        maxNum = temp;
                    }
                }
            }
        }
        cout << "Min num: " << int(minNum) << endl;
        cout << "Max num: " << int(maxNum) << endl;
    }
}

// hàm nhân ma trận
void multiply_two_matrix_host(unsigned char* matrix_a, unsigned char* matrix_b, unsigned char* dst_matrix,unsigned int n_rows_a, unsigned int n_cols_a, unsigned n_cols_b) {
    unsigned int required_mem_size = n_rows_a * n_cols_b * sizeof(unsigned char);
    dst_matrix = (unsigned char*)malloc(required_mem_size);

    for (int i = 0; i < n_rows_a; i++) {
        for (int j = 0; j < n_cols_b; j++) {
            unsigned char temp = 0;
            for (int k = 0; k < n_cols_a; k++) {
                temp += matrix_a[i*n_cols_a + k] * matrix_b[k*n_cols_b+j];
            }
            dst_matrix[i*n_cols_b+j] = temp;
        }
    }
}


// hàm softmax

// hàm lan truyền tiến

// hàm lan truyền ngược



int main() {
    string filename = "train-images-idx3-ubyte";
    unsigned int* input_data = NULL;
    unsigned int number_of_images, n_rows, n_cols;
    read_mnist(filename, input_data, number_of_images, n_rows, n_cols);
    unsigned long long temp = number_of_images * n_rows * n_cols;

    for (int i = 0; i < temp; i++) {
        if (input_data[i] != 0 && input_data[i] != 255) {
            cout << int(input_data[i]) << endl;
        }
    }
    free(input_data);
}