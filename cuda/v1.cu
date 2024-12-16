#include "common.hpp"

#include <fstream>
#include <cstdint>
#include <exception>
#include <vector>
#include <curand_kernel.h>

using namespace std;

int32_t reverseInt32(uint8_t bytes[]) {
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

uint8_t* readImagesIntoHostMemory(string& file_path, int& num_images, int& image_height, int& image_width) {
    ifstream file(file_path, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("cannot open file " + file_path);
    }

    uint8_t buffer[16];
    file.read(reinterpret_cast<char*>(buffer), 16);
    int32_t magic_number = reverseInt32(buffer);
    num_images = reverseInt32(buffer + 4);
    image_height = reverseInt32(buffer + 8);
    image_width = reverseInt32(buffer + 12);
    if (magic_number != 0x803) {
        throw runtime_error("file contains invalid format - magic number " + magic_number);
    }
    LOG("Found " << num_images << " images, size " << image_height << " x " << image_width << ".");

    int num_pixels_per_image = image_width * image_height;
    uint8_t* images = new uint8_t[num_images * num_pixels_per_image];

    for (int i = 0; i < num_images; ++i) {
        file.read(reinterpret_cast<char*>(&images[i * num_pixels_per_image]), num_pixels_per_image * sizeof(uint8_t));
    }

    file.close();
    return images;
}

uint8_t* readLabelsIntoHostMemory(string& file_path, int num_categories) {
    ifstream file(file_path, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("cannot open file " + file_path);
    }

    uint8_t buffer[8];
    file.read(reinterpret_cast<char*>(buffer), 8);
    int32_t magic_number = reverseInt32(buffer);
    int32_t num_labels = reverseInt32(buffer + 4);
    if (magic_number != 0x801) {
        throw runtime_error("file contains invalid format - magic number " + magic_number);
    }
    LOG("Found " << num_labels << " labels.");

    uint8_t* onehot_labels = new uint8_t[num_labels * num_categories];
    uint8_t* labels = new uint8_t[num_labels];
    fill(onehot_labels, onehot_labels + num_labels * num_categories, 0);
    file.read(reinterpret_cast<char*>(labels), num_labels * sizeof(uint8_t));
    for (int i = 0; i < num_labels; ++i) {
        onehot_labels[i * num_categories + labels[i]] = 1;
    }

    file.close();
    delete[] labels;
    return onehot_labels;
}

__global__ void g_transferAndConvertHTD(uint8_t* h_data, float* d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] = static_cast<float>(h_data[idx]);
    }
}

__global__ void g_randomizeValues(float* a, int n, unsigned long long seed = 666) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        a[idx] = curand_uniform(&state) * 2.0f - 1.0f;
    }
}

__global__ void g_mulMats(float* mat_a, float* mat_b, float* mat_out, int m, int n, int k) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    float out_rc = 0;
    if (r < m && c < k) {
        for (int i = 0; i < n; ++i) {
            out_rc += mat_a[r * n + i] * mat_b[i * k + c];
        }
        mat_out[r * k + c] = out_rc;
    }
}

__global__ void g_addRowsMatVec(float* mat_a, float* vec_b, int m, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < m && c < n) {
        mat_a[r * n + c] += vec_b[c];
    }
}

__global__ void g_activReLU(float* mat_in, float* mat_out, int m, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < m && c < n) {
        mat_out[r * n + c] = max(mat_in[r * n + c], 0.0f);
    }
}

__global__ void g_subRowsMats(float* mat_a, float* mat_b, float* mat_out, int m, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < m && c < n) {
        mat_out[r * n + c] = mat_a[r * n + c] - mat_b[r * n + c];
    } 
}

__global__ void g_mulMatsFirstTransposed(float* mat_a, float* mat_b, float* mat_out, int m, int n, int k) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    float out_rc = 0;
    if (r < m && c < k) {
        for (int i = 0; i < n; ++i) {
            out_rc += mat_a[i * m + r] * mat_b[i * k + c];
        }
        mat_out[r * k + c] = out_rc;
    }
}

__global__ void g_mulMatsSecondTransposed(float* mat_a, float* mat_b, float* mat_out, int m, int n, int k) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    float out_rc = 0;
    if (r < m && c < k) {
        for (int i = 0; i < n; ++i) {
            out_rc += mat_a[r * n + i] * mat_b[c * n + i];
        }
        mat_out[r * k + c] = out_rc;
    }
}

__global__ void g_sumColsMat(float* mat, float* vec_out, int m, int n) {
    int r = blockIdx.y * blockDim.y * 2 + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m || c >= n) return;

    float org_val_1 = mat[r * n + c];
    float org_val_2 = (r + blockDim.y < m) ? mat[(r + blockDim.y) * n + c] : 0;
    for (int stride = blockDim.y; stride >= 1; stride /= 2) {
        if (threadIdx.y < stride && r + stride < m) {
            mat[r * n + c] += mat[(r + stride) * n + c];
        }
        __syncthreads();
    }
    if (threadIdx.y == 0) {
        atomicAdd(&vec_out[c], mat[blockIdx.y * blockDim.y * 2]);
    }
    __syncthreads();
    mat[r * n + c] = org_val_1;
    if (r + blockDim.y < m) mat[(r + blockDim.y) * n + c] = org_val_2;
}

__global__ void g_mulMatsElemWise(float* mat_a, float* mat_b, float* mat_out, int m, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (r < m && c < n) {
        mat_out[r * n + c] = mat_a[r * n + c] * mat_b[r * n + c];
    }
}

__global__ void g_computeDerivReLU(float* mat_in, float* mat_out, int m, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < m && c < n) {
        mat_out[r * n + c] = (mat_in[r * n + c] > 0) ? 1 : 0;
    }
}

__global__ void g_addLinear(float* dst, float* amount, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += alpha * amount[idx];
    }
}


__global__ void g_matSoftmax(float* mat_in, float* mat_out, float* sums, int m, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m || c >= n) return;
    
    mat_out[r * n + c] = exp(mat_in[r * n + c]) / sums[r];
}

__global__ void crossEntropyKernel(float* y_pred, float* y_true, float* result, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float sumImage = 0.0;
        for (int j = 0; j < cols; j++) {
            int index = row * cols + j;
            sumImage += y_true[index] * log(y_pred[index]);
        }
        result[row] = -sumImage;
    }
}




__global__ void accuracyKernel(float* y_pred, float* y_true, int* correct, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        int predicted_label = 0;
        float max_val = y_pred[row * cols];
        for (int j = 1; j < cols; j++) {
            int index = row * cols + j;
            if (y_pred[index] > max_val) {
                max_val = y_pred[index];
                predicted_label = j;
            }
        }
        if (y_true[row * cols + predicted_label] == 1.0) {
            atomicAdd(correct, 1);
        }
    }
}


int main(int argc, char* argv[]) {
    int num_images, image_height, image_width;
    int num_epochs = 10;
    int num_categories = 10;
    float learning_rate = 0.001f;
    string train_data_path = "./train-images-idx3-ubyte";
    string train_label_path = "./train-labels-idx1-ubyte";

    // Read images and labels from file into host memory
    uint8_t* h_train_images = readImagesIntoHostMemory(train_data_path, num_images, image_height, image_width);
    uint8_t* h_train_labels = readLabelsIntoHostMemory(train_label_path, num_categories);
    LOG("Loaded training images and labels into host memory.");

    // Transfer data from host to device and convert them from integers to floats
    int num_pixels_per_image = image_height * image_width;
    float* h_train_images_pinned;
    float* h_train_labels_pinned;

    CHECK_CUDA(cudaMallocHost((void**)&h_train_images_pinned, num_images * num_pixels_per_image * sizeof(float)));
    CHECK_CUDA(cudaMallocHost((void**)&h_train_labels_pinned, num_images * num_pixels_per_image * sizeof(float)));

    for (int i = 0; i < num_images * num_pixels_per_image; ++i) {
        h_train_images_pinned[i] = static_cast<float>(h_train_images[i]);
    }
    for (int i = 0; i < num_images * num_categories; ++i) {
        h_train_labels_pinned[i] = static_cast<float>(h_train_labels[i]);
    }

    float* d_train_images;
    float* d_train_labels;
    CHECK_CUDA(cudaMalloc((void**)&d_train_images, num_images * num_pixels_per_image * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_train_labels, num_images * num_categories * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_train_images, h_train_images_pinned, num_images * num_pixels_per_image * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_train_labels, h_train_labels_pinned, num_images * num_categories * sizeof(float), cudaMemcpyHostToDevice));
    LOG("Training data transfered to device memory.");

    int n = num_images, n_0 = num_pixels_per_image, n_1 = 128, n_2 = 128, n_3 = num_categories;
    float* d_w_1;
    float* d_b_1;
    float* d_z_1;
    float* d_a_1;
    float* d_w_2;
    float* d_b_2;
    float* d_z_2;
    float* d_a_2;
    float* d_w_3;
    float* d_b_3;
    float* d_z_3;
    // float* d_a_3;

    float* d_grad_w_1;
    float* d_grad_b_1;
    float* d_grad_z_1;
    float* d_grad_a_1_z_1;
    float* d_grad_a_1;

    float* d_grad_w_2;
    float* d_grad_b_2;
    float* d_grad_z_2;
    float* d_grad_a_2_z_2;
    float* d_grad_a_2;
    
    float* d_grad_w_3; // n_2 x n_3
    float* d_grad_b_3; // 1 x n_3
    float* d_grad_z_3;

    float* d_sums;
    float* d_softmax;
    float* d_loss;

    int* accuracy;


    CHECK_CUDA(cudaMalloc((void**)&d_w_1, n_0 * n_1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_b_1, n_1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_z_1, n * n_1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_a_1, n * n_1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_w_2, n_1 * n_2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_b_2, n_2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_z_2, n * n_2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_a_2, n * n_2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_w_3, n_2 * n_3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_b_3, n_3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_z_3, n * n_3 * sizeof(float)));
    // CHECK_CUDA(cudaMalloc((void**)&d_a_3, n * n_3 * sizeof(float)));

    CHECK_CUDA(cudaDeviceSynchronize());
    LOG("Weights allocated.");
    CHECK_CUDA(cudaMalloc((void**)&d_grad_w_1, n_0 * n_1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_grad_b_1, n_1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_grad_z_1, n * n_1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_grad_a_1_z_1, n * n_1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_grad_a_1, n * n_1 * sizeof(float)));

    CHECK_CUDA(cudaMalloc((void**)&d_grad_w_2, n_1 * n_2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_grad_b_2, n_2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_grad_z_2, n * n_2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_grad_a_2_z_2, n * n_2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_grad_a_2, n * n_2 * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc((void**)&d_grad_w_3, n_2 * n_3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_grad_b_3, n_3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_grad_z_3, n * n_3 * sizeof(float)));

    CHECK_CUDA(cudaMalloc((void**)&d_sums, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_softmax, n * n_3 * sizeof(float)));
        
    CHECK_CUDA(cudaMalloc((void**)&d_loss, n * sizeof(float)));


    CHECK_CUDA(cudaMalloc((void**)&accuracy, sizeof(int)));
    // Randomize weights
    unsigned long long random_seed = 666;
    {
        dim3 block_size(DEFAULT_BLOCKSIZE);
        dim3 grid_size((n_0 * n_1 + block_size.x - 1) / block_size.x);
        g_randomizeValues<<<grid_size, block_size>>>(d_w_1, n_0 * n_1, random_seed);
    }
    {
        dim3 block_size(DEFAULT_BLOCKSIZE);
        dim3 grid_size((n_1 + block_size.x - 1) / block_size.x);
        g_randomizeValues<<<grid_size, block_size>>>(d_b_1, n_1, random_seed);
    }
    {
        dim3 block_size(DEFAULT_BLOCKSIZE);
        dim3 grid_size((n_1 * n_2 + block_size.x - 1) / block_size.x);
        g_randomizeValues<<<grid_size, block_size>>>(d_w_2, n_1 * n_2, random_seed);
    }
    {
        dim3 block_size(DEFAULT_BLOCKSIZE);
        dim3 grid_size((n_2 + block_size.x - 1) / block_size.x);
        g_randomizeValues<<<grid_size, block_size>>>(d_b_2, n_2, random_seed);
    }
    {
        dim3 block_size(DEFAULT_BLOCKSIZE);
        dim3 grid_size((n_2 * n_3 + block_size.x - 1) / block_size.x);
        g_randomizeValues<<<grid_size, block_size>>>(d_w_3, n_2 * n_3, random_seed);
    }
    {
        dim3 block_size(DEFAULT_BLOCKSIZE);
        dim3 grid_size((n_3 + block_size.x - 1) / block_size.x);
        g_randomizeValues<<<grid_size, block_size>>>(d_b_3, n_3, random_seed);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    LOG("Weights randomized.");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Forward
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_1 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
            g_mulMats<<<grid_size, block_size>>>(d_train_images, d_w_1, d_z_1, n, n_0, n_1);
            g_addRowsMatVec<<<grid_size, block_size>>>(d_z_1, d_b_1, n, n_1);
            g_activReLU<<<grid_size, block_size>>>(d_z_1, d_a_1, n, n_1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("Forwarded layer 1.");
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_2 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
            g_mulMats<<<grid_size, block_size>>>(d_a_1, d_w_2, d_z_1, n, n_1, n_2);
            g_addRowsMatVec<<<grid_size, block_size>>>(d_z_2, d_b_2, n, n_2);
            g_activReLU<<<grid_size, block_size>>>(d_z_2, d_a_2, n, n_2);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("Forwarded layer 2.");
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_3 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
            g_mulMats<<<grid_size, block_size>>>(d_a_2, d_w_3, d_z_3, n, n_2, n_3);
            g_addRowsMatVec<<<grid_size, block_size>>>(d_z_3, d_b_3, n, n_3);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("Forwarded layer 3.");

        // Softmax
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_3 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
            g_sumColsMat<<<grid_size, block_size>>>(d_z_3, d_sums, n, n_3);
        }
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_3 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
            g_matSoftmax<<<grid_size, block_size>>>(d_z_3, d_softmax, d_sums, n, n_3);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("Calculate softmax");

        // Backward
        // L / z3
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_3 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
            g_subRowsMats<<<grid_size, block_size>>>(d_z_3, d_train_labels, d_grad_z_3, n, n_3);
        }
        
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("L/z3");
        // L / w3
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_3 + block_size.x - 1) / block_size.x, (n_2 + block_size.y - 1) / block_size.y);
            g_mulMatsFirstTransposed<<<grid_size, block_size>>>(d_a_2, d_grad_z_3, d_grad_w_3, n_2, n, n_3);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("L/w3");
        // L / b3
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_3 + block_size.x * 2 - 1) / (block_size.x * 2), (n + block_size.y - 1) / block_size.y);
            g_sumColsMat<<<grid_size, block_size>>>(d_z_3, d_grad_b_3, n, n_3);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("L/b3");
        // L / a2
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_2 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y); 
            g_mulMatsSecondTransposed<<<grid_size, block_size>>>(d_grad_z_3, d_w_3, d_grad_a_2, n, n_3, n_2);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("L/a2");
        // a2 / z2
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_2 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
            g_computeDerivReLU<<<grid_size, block_size>>>(d_a_2, d_grad_a_2_z_2, n, n_2);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("a2/z2");
        // L / z2
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_2 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
            g_mulMatsElemWise<<<grid_size, block_size>>>(d_grad_a_2, d_grad_a_2_z_2, d_grad_z_2, n, n_2);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("L/z2");
        // L / w2
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_2 + block_size.x - 1) / block_size.x, (n_1 + block_size.y - 1) / block_size.y);
            g_mulMatsFirstTransposed<<<grid_size, block_size>>>(d_a_1, d_grad_z_2, d_grad_w_2, n_1, n, n_2);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("L/w2");
        // L / b2
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_2 + block_size.x * 2 - 1) / (block_size.x * 2), (n + block_size.y - 1) / block_size.y);
            g_sumColsMat<<<grid_size, block_size>>>(d_z_2, d_grad_b_2, n, n_2);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("L/b2");
        // L / a1
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_1 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y); 
            g_mulMatsSecondTransposed<<<grid_size, block_size>>>(d_grad_z_2, d_w_2, d_grad_a_1, n, n_2, n_1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("L/a1");
        // L / z1
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_1 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
            g_mulMatsElemWise<<<grid_size, block_size>>>(d_grad_a_1, d_grad_a_1_z_1, d_grad_z_1, n, n_1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("L/z1");
        // L / w1
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_1 + block_size.x - 1) / block_size.x, (n_0 + block_size.y - 1) / block_size.y);
            g_mulMatsFirstTransposed<<<grid_size, block_size>>>(d_train_images, d_grad_z_1, d_grad_w_1, n_0, n, n_1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("L/w1");
        // L / b1
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_1 + block_size.x * 2 - 1) / (block_size.x * 2), (n + block_size.y - 1) / block_size.y);
            g_sumColsMat<<<grid_size, block_size>>>(d_z_1, d_grad_b_1, n, n_1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("L/b1");

        // Update weight
        // w1
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_1 + block_size.x - 1) / block_size.x, (n_0 + block_size.y - 1) / block_size.y);
            g_addLinear<<<grid_size, block_size>>>(d_w_1, d_grad_w_1, -learning_rate, n * n_1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("Update w1");
        // b1
        {
            dim3 block_size(DEFAULT_BLOCKSIZE);
            dim3 grid_size((n_1 + block_size.x - 1) / block_size.x);
            g_addLinear<<<grid_size, block_size>>>(d_b_1, d_grad_b_1, -learning_rate, n_1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("Update b1");
        // w2
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_2 + block_size.x - 1) / block_size.x, (n_1 + block_size.y - 1) / block_size.y);
            g_addLinear<<<grid_size, block_size>>>(d_w_2, d_grad_w_2, -learning_rate, n * n_2);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("Update w2");
        // b2
        {
            dim3 block_size(DEFAULT_BLOCKSIZE);
            dim3 grid_size((n_2 + block_size.x - 1) / block_size.x);
            g_addLinear<<<grid_size, block_size>>>(d_b_2, d_grad_b_2, -learning_rate, n_2);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("Update b2");
        // w3
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_2 + block_size.x - 1) / block_size.x, (n_1 + block_size.y - 1) / block_size.y);
            g_addLinear<<<grid_size, block_size>>>(d_w_2, d_grad_w_2, -learning_rate, n * n_2);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("Update w3");
        // b3
        {
            dim3 block_size(DEFAULT_BLOCKSIZE);
            dim3 grid_size((n_3 + block_size.x - 1) / block_size.x);
            g_addLinear<<<grid_size, block_size>>>(d_b_3, d_grad_b_3, -learning_rate, n_3);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        LOG("Update b3");
        
        //compute loss
        {
            float loss = 0.0f;
            float *h_loss;
            CHECK_CUDA(cudaMallocHost((void**)&h_loss, n * sizeof(float)));
            dim3 block_size(DEFAULT_BLOCKSIZE);
            dim3 grid_size((n + block_size.x - 1) / block_size.x);
            crossEntropyKernel<<<grid_size, block_size>>>(d_softmax, d_train_labels, d_loss, n, n_3);
            CHECK_CUDA(cudaMemcpy(h_loss, d_loss, n * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaDeviceSynchronize());

            for (int i = 0; i < n; ++i) {
                loss += h_loss[i];
            }
            LOG("Epoch " << epoch << " completed. Loss: " << loss);
        }

        //compute accuracy
        {
            float acc = 0.0f;
            int *h_accuracy;
            CHECK_CUDA(cudaMallocHost((void**)&h_accuracy, sizeof(int)));
            dim3 block_size(DEFAULT_BLOCKSIZE);
            dim3 grid_size((n + block_size.x - 1) / block_size.x);
            accuracyKernel<<<grid_size, block_size>>>(d_softmax, d_train_labels, accuracy, n, n_3);
            CHECK_CUDA(cudaMemcpy(h_accuracy, accuracy, sizeof(int), cudaMemcpyDeviceToHost));
            acc = static_cast<float>(*h_accuracy) / n;
            LOG("Epoch " << epoch << " completed. Accuracy: " << acc);

        }

        CHECK_CUDA(cudaDeviceSynchronize());
        printf("Epoch %d completed.\n", epoch);
    }

    CHECK_CUDA(cudaFree(d_w_1));
    CHECK_CUDA(cudaFree(d_b_1));
    CHECK_CUDA(cudaFree(d_z_1));
    CHECK_CUDA(cudaFree(d_a_1));
    CHECK_CUDA(cudaFree(d_w_2));
    CHECK_CUDA(cudaFree(d_b_2));
    CHECK_CUDA(cudaFree(d_z_2));
    CHECK_CUDA(cudaFree(d_a_2));
    CHECK_CUDA(cudaFree(d_w_3));
    CHECK_CUDA(cudaFree(d_b_3));
    CHECK_CUDA(cudaFree(d_z_3));

    CHECK_CUDA(cudaFree(d_grad_w_1));
    CHECK_CUDA(cudaFree(d_grad_b_1));
    CHECK_CUDA(cudaFree(d_grad_z_1));
    CHECK_CUDA(cudaFree(d_grad_a_1_z_1));
    CHECK_CUDA(cudaFree(d_grad_a_1));
    CHECK_CUDA(cudaFree(d_grad_w_2));
    CHECK_CUDA(cudaFree(d_grad_b_2));
    CHECK_CUDA(cudaFree(d_grad_z_2));
    CHECK_CUDA(cudaFree(d_grad_a_2_z_2));
    CHECK_CUDA(cudaFree(d_grad_a_2));
    CHECK_CUDA(cudaFree(d_grad_w_3));
    CHECK_CUDA(cudaFree(d_grad_b_3));
    CHECK_CUDA(cudaFree(d_grad_z_3));
    CHECK_CUDA(cudaFree(d_loss));
    CHECK_CUDA(cudaFree(d_sums));
    CHECK_CUDA(cudaFree(d_softmax));
    CHECK_CUDA(cudaFreeHost(h_train_images_pinned));
    CHECK_CUDA(cudaFreeHost(h_train_labels_pinned));

    CHECK_CUDA(cudaFree(d_train_images));
    CHECK_CUDA(cudaFree(d_train_labels));

    delete[] h_train_images;
    delete[] h_train_labels;


    return 0;
}