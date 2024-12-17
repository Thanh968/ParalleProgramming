#include "common.hpp"

#include <fstream>
#include <cstdint>
#include <exception>
#include <vector>
#include <curand_kernel.h>

using namespace std;

bool infer_mode = false;
string train_images_path;
string train_labels_path;
string val_images_path;
string val_labels_path;
string save_weights_path;
string load_weights_path;
int num_train_images;
int num_val_images;

int num_epochs = 10;
constexpr float learning_rate = 1e-6f;

constexpr int image_height = 28;
constexpr int image_width = 28;
constexpr int num_categories = 10;
constexpr int num_pixels_per_image = image_height * image_width;
constexpr int n_0 = num_pixels_per_image;
constexpr int n_1 = 128;
constexpr int n_2 = 128;
constexpr int n_3 = num_categories;
constexpr int num_weights = n_0 * n_1 + n_1 + n_1 * n_2 + n_2 + n_2 * n_3 + n_3;
constexpr int offset_w_1 = 0;
constexpr int offset_b_1 = offset_w_1 + n_0 * n_1;
constexpr int offset_w_2 = offset_b_1 + n_1;
constexpr int offset_b_2 = offset_w_2 + n_1 * n_2;
constexpr int offset_w_3 = offset_b_2 + n_2;
constexpr int offset_b_3 = offset_w_3 + n_2 * n_3;

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
float* d_a_3;
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
float* d_grad_w_3;
float* d_grad_b_3;
float* d_grad_z_3;

float* d_z_1_infer;
float* d_a_1_infer;
float* d_z_2_infer;
float* d_a_2_infer;
float* d_z_3_infer;
float* d_a_3_infer;

float* h_loss_train;
float* d_loss_train;
int* h_count_correct_train;
int* d_count_correct_train;
float* h_loss_infer;
float* d_loss_infer;
int* h_count_correct_infer;
int* d_count_correct_infer;

void parseArguments(int argc, char* argv[]) {
    int i = 1;
    while (i < argc) {
        if (strcmp(argv[i], "--infer") == 0 || strcmp(argv[i], "-i") == 0) {
            infer_mode = true;
            i += 1;
        } else if (strcmp(argv[i], "--train-images") == 0) {
            train_images_path = argv[i + 1];
            i += 2;
        } else if (strcmp(argv[i], "--train-labels") == 0) {
            train_labels_path = argv[i + 1];
            i += 2;
        } else if (strcmp(argv[i], "--val-images") == 0) {
            val_images_path = argv[i + 1];
            i += 2;
        } else if (strcmp(argv[i], "--val-labels") == 0) {
            val_labels_path = argv[i + 1];
            i += 2;
        }else if (strcmp(argv[i], "--save-checkpoint") == 0) {
            save_weights_path = argv[i + 1];
            i += 2;
        } else if (strcmp(argv[i], "--load-checkpoint") == 0) {
            load_weights_path = argv[i + 1];
            i += 2;
        } else if (strcmp(argv[i], "--num-epochs") == 0) {
            num_epochs = atoi(argv[i + 1]);
            i += 2;
        } else {
            throw runtime_error("invalid arguments");
        }
    }
}

int32_t reverseInt32(uint8_t bytes[]) {
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

uint8_t* readImagesIntoHostMemory(string& file_path, int& num_images) {
    ifstream file(file_path, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("cannot open file " + file_path);
    }

    uint8_t buffer[16];
    file.read(reinterpret_cast<char*>(buffer), 16);
    int32_t magic_number = reverseInt32(buffer);
    num_images = reverseInt32(buffer + 4);
    int32_t read_image_height = reverseInt32(buffer + 8);
    int32_t read_image_width = reverseInt32(buffer + 12);
    if (magic_number != 0x803) {
        throw runtime_error("file contains invalid format - magic number " + magic_number);
    }
    if (read_image_height != image_height || read_image_width != image_width) {
        throw runtime_error("unexpected image size");
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

uint8_t* readLabelsIntoHostMemory(string& file_path) {
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

void initData(string images_path, string labels_path, float*& d_images, float*& d_labels, int& num_images) {
    uint8_t* h_images = readImagesIntoHostMemory(images_path, num_images);
    uint8_t* h_labels = readLabelsIntoHostMemory(labels_path);

    float* h_images_pinned;
    float* h_labels_pinned;
    CHECK_CUDA(cudaMallocHost((void**)&h_images_pinned, num_images * num_pixels_per_image * sizeof(float)));
    CHECK_CUDA(cudaMallocHost((void**)&h_labels_pinned, num_images * num_categories * sizeof(float)));
    for (int i = 0; i < num_images * num_pixels_per_image; ++i) {
        h_images_pinned[i] = static_cast<float>(h_images[i]) / 255.0f;
    }
    for (int i = 0; i < num_images * num_categories; ++i) {
        h_labels_pinned[i] = static_cast<float>(h_labels[i]);
    }

    CHECK_CUDA(cudaMalloc((void**)&d_images, num_images * num_pixels_per_image * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_labels, num_images * num_categories * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_images, h_images_pinned, num_images * num_pixels_per_image * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_labels, h_labels_pinned, num_images * num_categories * sizeof(float), cudaMemcpyHostToDevice));
    LOG("Training data transfered to device memory.");

    CHECK_CUDA(cudaFreeHost(h_images_pinned));
    CHECK_CUDA(cudaFreeHost(h_labels_pinned));

    delete[] h_images;
    delete[] h_labels;
}

void saveWeights(string file_path) {
    ofstream file(file_path, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("cannot open file " + file_path);
    }

    float* h_weights;
    CHECK_CUDA(cudaMallocHost((void**)&h_weights, num_weights * sizeof(float)));

    CHECK_CUDA(cudaMemcpy((void*)(h_weights + offset_w_1), d_w_1, n_0 * n_1 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy((void*)(h_weights + offset_b_1), d_b_1, n_1 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy((void*)(h_weights + offset_w_2), d_w_2, n_1 * n_2 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy((void*)(h_weights + offset_b_2), d_b_2, n_2 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy((void*)(h_weights + offset_w_3), d_w_3, n_2 * n_3 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy((void*)(h_weights + offset_b_3), d_b_3, n_3 * sizeof(float), cudaMemcpyDeviceToHost));

    file.write(reinterpret_cast<char*>(h_weights), num_weights * sizeof(float));
    LOG("Weights saved to file.");

    CHECK_CUDA(cudaFreeHost(h_weights));
    file.close();
}

void loadWeights(string file_path) {
    ifstream file(file_path, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("cannot open file " + file_path);
    }

    float* h_weights;
    CHECK_CUDA(cudaMallocHost((void**)&h_weights, num_weights * sizeof(float)));

    if (!file.read(reinterpret_cast<char*>(h_weights), num_weights * sizeof(float))) {
        CHECK_CUDA(cudaFree(h_weights));
        file.close();
        throw runtime_error("cannot read file " + file_path);
    }
    LOG("Weights loaded from file into host memory.");

    CHECK_CUDA(cudaMemcpy(d_w_1, (void*)(h_weights + offset_w_1), n_0 * n_1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_1, (void*)(h_weights + offset_b_1), n_1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w_2, (void*)(h_weights + offset_w_2), n_1 * n_2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_2, (void*)(h_weights + offset_b_2), n_2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w_3, (void*)(h_weights + offset_w_3), n_2 * n_3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_3, (void*)(h_weights + offset_b_3), n_3 * sizeof(float), cudaMemcpyHostToDevice));
    LOG("Weights transfered from host to device memory.");

    CHECK_CUDA(cudaFreeHost(h_weights));
    file.close();
}

__global__ void g_transferAndConvertHTD(uint8_t* h_data, float* d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] = static_cast<float>(h_data[idx]);
    }
}

__global__ void g_heWeightInitialization(float* d_weights, int m, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = m * n;
    if (idx >= total_size) return;

    curandState state;
    curand_init(seed, idx, 0, &state);
    float random_normal = curand_normal(&state);
    float stddev = sqrtf(2.0f / (float)m);
    d_weights[idx] = random_normal * stddev;
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

__global__ void g_activSoftmax(float* mat_in, float* mat_out, int m, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= m) return;

    float maxVal = -INFINITY;
    float sumExp = 0.0;

    for (int c = 0; c < n; c++) {
        maxVal = fmaxf(maxVal, mat_in[r * n + c]);
    }

    for (int c = 0; c < n; c++) {
        mat_out[r * n + c] = expf(mat_in[r * n + c] - maxVal);
        sumExp += mat_out[r * n + c];
    }

    for (int c = 0; c < n; c++) {
        mat_out[r * n + c] /= sumExp;
        mat_out[r * n + c] = max(mat_out[r * n + c], 0.001f);
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

// __global__ void g_sumColsMat(float* mat, float* vec_out, int m, int n) {
//     int r = blockIdx.y * blockDim.y * 2 + threadIdx.y;
//     int c = blockIdx.x * blockDim.x + threadIdx.x;
//     if (r >= m || c >= n) return;

//     float org_val_1 = mat[r * n + c];
//     float org_val_2 = (r + blockDim.y < m) ? mat[(r + blockDim.y) * n + c] : 0;
//     for (int stride = blockDim.y; stride >= 1; stride /= 2) {
//         if (threadIdx.y < stride && r + stride < m) {
//             mat[r * n + c] += mat[(r + stride) * n + c];
//         }
//         __syncthreads();
//     }
//     if (threadIdx.y == 0) {
//         atomicAdd(&vec_out[c], mat[blockIdx.y * blockDim.y * 2 * n]);
//     }
//     __syncthreads();
//     mat[r * n + c] = org_val_1;
//     if (r + blockDim.y < m) mat[(r + blockDim.y) * n + c] = org_val_2;
// }

__global__ void g_sumColsMat(float* mat, float* vec_out, int m, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;

    float sum = 0.0f;
    for (int i = 0; i < m; ++i) {
        sum += mat[i * n + c];
    }
    vec_out[c] = sum;
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

__global__ void g_computeCrossEntropy(float* y_pred, float* y_true, float* result, int rows, int cols) {
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

__global__ void g_computeAccuracy(float* y_pred, float* y_true, int* correct, int rows, int cols) {
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

void print(float* d_data, int m, int n) {
    float* h_data;
    CHECK_CUDA(cudaMallocHost((void**)&h_data, m * n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(h_data, d_data, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", h_data[i * n + j]);
        }
        printf("\n");
    }
    CHECK_CUDA(cudaFreeHost(h_data));
}

void train() {
    float* d_train_images;
    float* d_train_labels;
    float* d_val_images;
    float* d_val_labels;

    initData(train_images_path, train_labels_path, d_train_images, d_train_labels, num_train_images);
    initData(val_images_path, val_labels_path, d_val_images, d_val_labels, num_val_images);
    LOG("Data initialized.");
    
    int n = num_train_images, n_infer = num_val_images;
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
    CHECK_CUDA(cudaMalloc((void**)&d_a_3, n * n_3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_z_1_infer, n_infer * n_1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_a_1_infer, n_infer * n_1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_z_2_infer, n_infer * n_2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_a_2_infer, n_infer * n_2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_z_3_infer, n_infer * n_3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_a_3_infer, n_infer * n_3 * sizeof(float)));

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

    CHECK_CUDA(cudaMallocHost((void**)&h_loss_train, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_loss_train, n * sizeof(float)));
    CHECK_CUDA(cudaMallocHost((void**)&h_count_correct_train, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_count_correct_train, sizeof(int)));
    CHECK_CUDA(cudaMallocHost((void**)&h_loss_infer, n_infer * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_loss_infer, n_infer * sizeof(float)));
    CHECK_CUDA(cudaMallocHost((void**)&h_count_correct_infer, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_count_correct_infer, sizeof(int)));

    unsigned long long random_seed = 666;
    {
        dim3 block_size(DEFAULT_BLOCKSIZE);
        dim3 grid_size((n_0 * n_1 + block_size.x - 1) / block_size.x);
        // g_randomizeValues<<<grid_size, block_size>>>(d_w_1, n_0 * n_1, random_seed);
        g_heWeightInitialization<<<grid_size, block_size>>>(d_w_1, n_0, n_1, random_seed);
    }
    {
        dim3 block_size(DEFAULT_BLOCKSIZE);
        dim3 grid_size((n_1 + block_size.x - 1) / block_size.x);
        g_randomizeValues<<<grid_size, block_size>>>(d_b_1, n_1, random_seed);
    }
    {
        dim3 block_size(DEFAULT_BLOCKSIZE);
        dim3 grid_size((n_1 * n_2 + block_size.x - 1) / block_size.x);
        // g_randomizeValues<<<grid_size, block_size>>>(d_w_2, n_1 * n_2, random_seed);
        g_heWeightInitialization<<<grid_size, block_size>>>(d_w_2, n_1, n_2, random_seed);

    }
    {
        dim3 block_size(DEFAULT_BLOCKSIZE);
        dim3 grid_size((n_2 + block_size.x - 1) / block_size.x);
        g_randomizeValues<<<grid_size, block_size>>>(d_b_2, n_2, random_seed);
    }
    {
        dim3 block_size(DEFAULT_BLOCKSIZE);
        dim3 grid_size((n_2 * n_3 + block_size.x - 1) / block_size.x);
        // g_randomizeValues<<<grid_size, block_size>>>(d_w_3, n_2 * n_3, random_seed);
        g_heWeightInitialization<<<grid_size, block_size>>>(d_w_3, n_2, n_3, random_seed);
    }
    {
        dim3 block_size(DEFAULT_BLOCKSIZE);
        dim3 grid_size((n_3 + block_size.x - 1) / block_size.x);
        g_randomizeValues<<<grid_size, block_size>>>(d_b_3, n_3, random_seed);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    LOG("Weights initialized.");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Forward
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_1 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
            g_mulMats<<<grid_size, block_size>>>(d_train_images, d_w_1, d_z_1, n, n_0, n_1);
            g_addRowsMatVec<<<grid_size, block_size>>>(d_z_1, d_b_1, n, n_1);
            g_activReLU<<<grid_size, block_size>>>(d_z_1, d_a_1, n, n_1);
        }
        BREAK;
        LOG("Forwarded layer 1.");
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_2 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
            g_mulMats<<<grid_size, block_size>>>(d_a_1, d_w_2, d_z_2, n, n_1, n_2);
            g_addRowsMatVec<<<grid_size, block_size>>>(d_z_2, d_b_2, n, n_2);
            g_activReLU<<<grid_size, block_size>>>(d_z_2, d_a_2, n, n_2);
            BREAK; print(d_a_2, 2, n_2);
        }
        BREAK;
        LOG("Forwarded layer 2.");
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_3 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
            g_mulMats<<<grid_size, block_size>>>(d_a_2, d_w_3, d_z_3, n, n_2, n_3);
            g_addRowsMatVec<<<grid_size, block_size>>>(d_z_3, d_b_3, n, n_3);
        }
        BREAK;
        {
            dim3 block_size(1, DEFAULT_BLOCKSIZE);
            dim3 grid_size(1, (n + block_size.y - 1) / block_size.y);
            g_activSoftmax<<<grid_size, block_size>>>(d_z_3, d_a_3, n, n_3);
        }
        BREAK;
        print(d_z_3 + (n - 18) * n_3, 18, n_3);
        printf("---\n");
        print(d_a_3 + (n - 18) * n_3, 18, n_3);
        LOG("Forwarded layer 3.");
        // compute loss
        {
            float loss = 0.0f;
            dim3 block_size(DEFAULT_BLOCKSIZE);
            dim3 grid_size((n + block_size.x - 1) / block_size.x);
            g_computeCrossEntropy<<<grid_size, block_size>>>(d_a_3, d_train_labels, d_loss_train, n, n_3);
            CHECK_CUDA(cudaMemcpy(h_loss_train, d_loss_train, n * sizeof(float), cudaMemcpyDeviceToHost));
            print(d_loss_train + n - 18, 18, 1);
            for (int i = 0; i < n; ++i) {
                loss += h_loss_train[i];
            }
            LOG("Epoch " << epoch << " completed. Train Loss: " << loss);
        }

        //compute accuracy
        {
            float acc = 0.0f;
            CHECK_CUDA(cudaMemset(d_count_correct_train, 0, sizeof(int)));
            dim3 block_size(DEFAULT_BLOCKSIZE);
            dim3 grid_size((n + block_size.x - 1) / block_size.x);
            g_computeAccuracy<<<grid_size, block_size>>>(d_z_3, d_train_labels, d_count_correct_train, n, n_3);
            CHECK_CUDA(cudaMemcpy(h_count_correct_train, d_count_correct_train, sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaDeviceSynchronize());
            acc = static_cast<float>(*h_count_correct_train) / n;
            LOG("Epoch " << epoch << " completed. Train Accuracy: " << acc);
        }

        // -------------------------------
        // Backward
        // L / z3
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_3 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
            g_subRowsMats<<<grid_size, block_size>>>(d_a_3, d_train_labels, d_grad_z_3, n, n_3);
        }
        // CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("L/z3");
        // L / w3
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_3 + block_size.x - 1) / block_size.x, (n_2 + block_size.y - 1) / block_size.y);
            g_mulMatsFirstTransposed<<<grid_size, block_size>>>(d_a_2, d_grad_z_3, d_grad_w_3, n_2, n, n_3);
        }
        // CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        print(d_a_2, 1, n_2);
        print(d_grad_w_3, 1, n_3);
        LOG("L/w3");
        // L / b3
        // {
        //     dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
        //     dim3 grid_size((n_3 + block_size.x - 1) / block_size.x, (n + block_size.y * 2 - 1) / (block_size.y * 2));
        //     g_sumColsMat<<<grid_size, block_size>>>(d_grad_z_3, d_grad_b_3, n, n_3);
        // }
        {
            dim3 block_size(DEFAULT_BLOCKSIZE);
            dim3 grid_size((n_3 + block_size.x - 1) / block_size.x);
            g_sumColsMat<<<grid_size, block_size>>>(d_grad_z_3, d_grad_b_3, n, n_3);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("L/b3");
        // L / a2
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_2 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y); 
            g_mulMatsSecondTransposed<<<grid_size, block_size>>>(d_grad_z_3, d_w_3, d_grad_a_2, n, n_3, n_2);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("L/a2");
        // a2 / z2
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_2 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
            g_computeDerivReLU<<<grid_size, block_size>>>(d_a_2, d_grad_a_2_z_2, n, n_2);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("a2/z2");
        // L / z2
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_2 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
            g_mulMatsElemWise<<<grid_size, block_size>>>(d_grad_a_2, d_grad_a_2_z_2, d_grad_z_2, n, n_2);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("L/z2");
        // L / w2
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_2 + block_size.x - 1) / block_size.x, (n_1 + block_size.y - 1) / block_size.y);
            g_mulMatsFirstTransposed<<<grid_size, block_size>>>(d_a_1, d_grad_z_2, d_grad_w_2, n_1, n, n_2);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("L/w2");
        // L / b2
        // {
        //     dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
        //     dim3 grid_size((n_2 + block_size.x - 1) / block_size.x, (n + block_size.y * 2 - 1) / (block_size.y * 2));
        //     g_sumColsMat<<<grid_size, block_size>>>(d_grad_z_2, d_grad_b_2, n, n_2);
        // }
        {
            dim3 block_size(DEFAULT_BLOCKSIZE);
            dim3 grid_size((n_2 + block_size.x - 1) / block_size.x);
            g_sumColsMat<<<grid_size, block_size>>>(d_grad_z_2, d_grad_b_2, n, n_2);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("L/b2");
        // L / a1
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_1 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y); 
            g_mulMatsSecondTransposed<<<grid_size, block_size>>>(d_grad_z_2, d_w_2, d_grad_a_1, n, n_2, n_1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("L/a1");
        // a1 / z1
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_1 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
            g_computeDerivReLU<<<grid_size, block_size>>>(d_a_1, d_grad_a_1_z_1, n, n_1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("a1/z1");
        // L / z1
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_1 + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
            g_mulMatsElemWise<<<grid_size, block_size>>>(d_grad_a_1, d_grad_a_1_z_1, d_grad_z_1, n, n_1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("L/z1");
        // L / w1
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_1 + block_size.x - 1) / block_size.x, (n_0 + block_size.y - 1) / block_size.y);
            g_mulMatsFirstTransposed<<<grid_size, block_size>>>(d_train_images, d_grad_z_1, d_grad_w_1, n_0, n, n_1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("L/w1");
        // L / b1
        // {
        //     dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
        //     dim3 grid_size((n_1 + block_size.x - 1) / block_size.x, (n + block_size.y * 2 - 1) / (block_size.y * 2));
        //     g_sumColsMat<<<grid_size, block_size>>>(d_grad_z_1, d_grad_b_1, n, n_1);
        // }
        {
            dim3 block_size(DEFAULT_BLOCKSIZE);
            dim3 grid_size((n_1 + block_size.x - 1) / block_size.x);
            g_sumColsMat<<<grid_size, block_size>>>(d_grad_z_1, d_grad_b_1, n, n_1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("L/b1");

        // Update weight
        // w1
        {
            dim3 block_size(DEFAULT_BLOCKSIZE);
            dim3 grid_size((n_0 * n_1 + block_size.x - 1) / block_size.x);
            g_addLinear<<<grid_size, block_size>>>(d_w_1, d_grad_w_1, -learning_rate, n_0 * n_1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("Update w1");
        // b1
        {
            dim3 block_size(DEFAULT_BLOCKSIZE);
            dim3 grid_size((n_1 + block_size.x - 1) / block_size.x);
            g_addLinear<<<grid_size, block_size>>>(d_b_1, d_grad_b_1, -learning_rate, n_1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("Update b1");
        // w2
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_1 * n_2 + block_size.x - 1) / block_size.x);
            g_addLinear<<<grid_size, block_size>>>(d_w_2, d_grad_w_2, -learning_rate, n_1 * n_2);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("Update w2");
        // b2
        {
            dim3 block_size(DEFAULT_BLOCKSIZE);
            dim3 grid_size((n_2 + block_size.x - 1) / block_size.x);
            g_addLinear<<<grid_size, block_size>>>(d_b_2, d_grad_b_2, -learning_rate, n_2);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("Update b2");
        // w3
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_2 * n_3 + block_size.x - 1) / block_size.x);
            g_addLinear<<<grid_size, block_size>>>(d_w_3, d_grad_w_3, -learning_rate, n_2 * n_3);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("Update w3");
        // b3
        {
            dim3 block_size(DEFAULT_BLOCKSIZE);
            dim3 grid_size((n_3 + block_size.x - 1) / block_size.x);
            g_addLinear<<<grid_size, block_size>>>(d_b_3, d_grad_b_3, -learning_rate, n_3);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("Update b3");

        // Forward
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_1 + block_size.x - 1) / block_size.x, (n_infer + block_size.y - 1) / block_size.y);
            g_mulMats<<<grid_size, block_size>>>(d_val_images, d_w_1, d_z_1_infer, n_infer, n_0, n_1);
            g_addRowsMatVec<<<grid_size, block_size>>>(d_z_1_infer, d_b_1, n_infer, n_1);
            g_activReLU<<<grid_size, block_size>>>(d_z_1_infer, d_a_1_infer, n_infer, n_1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("Forwarded layer 1.");
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_2 + block_size.x - 1) / block_size.x, (n_infer + block_size.y - 1) / block_size.y);
            g_mulMats<<<grid_size, block_size>>>(d_a_1_infer, d_w_2, d_z_2_infer, n_infer, n_1, n_2);
            g_addRowsMatVec<<<grid_size, block_size>>>(d_z_2_infer, d_b_2, n_infer, n_2);
            g_activReLU<<<grid_size, block_size>>>(d_z_2_infer, d_a_2_infer, n_infer, n_2);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("Forwarded layer 2.");
        {
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((n_3 + block_size.x - 1) / block_size.x, (n_infer + block_size.y - 1) / block_size.y);
            g_mulMats<<<grid_size, block_size>>>(d_a_2_infer, d_w_3, d_z_3_infer, n_infer, n_2, n_3);
            g_addRowsMatVec<<<grid_size, block_size>>>(d_z_3_infer, d_b_3, n_infer, n_3);
        }
        {
            dim3 block_size(1, DEFAULT_BLOCKSIZE);
            dim3 grid_size(1, (n_infer + block_size.y - 1) / block_size.y);
            g_activSoftmax<<<grid_size, block_size>>>(d_z_3_infer, d_a_3_infer, n_infer, n_3);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        BREAK;
        LOG("Forwarded layer 3.");

        // compute loss
        {
            float loss = 0.0f;
            dim3 block_size(DEFAULT_BLOCKSIZE);
            dim3 grid_size((n_infer + block_size.x - 1) / block_size.x);
            g_computeCrossEntropy<<<grid_size, block_size>>>(d_a_3_infer, d_val_labels, d_loss_infer, n_infer, n_3);
            CHECK_CUDA(cudaMemcpy(h_loss_infer, d_loss_infer, n_infer * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i) {
                loss += h_loss_infer[i];
            }
            LOG("Epoch " << epoch << " completed. Loss: " << loss);
        }

        //compute accuracy
        {
            float acc = 0.0f;
            CHECK_CUDA(cudaMemset(d_count_correct_infer, 0, sizeof(int)));
            dim3 block_size(DEFAULT_BLOCKSIZE);
            dim3 grid_size((n_infer + block_size.x - 1) / block_size.x);
            g_computeAccuracy<<<grid_size, block_size>>>(d_a_3_infer, d_val_labels, d_count_correct_infer, n_infer, n_3);
            CHECK_CUDA(cudaMemcpy(h_count_correct_infer, d_count_correct_infer, sizeof(int), cudaMemcpyDeviceToHost));
            acc = static_cast<float>(*h_count_correct_infer) / n_infer;
            LOG("Epoch " << epoch << " completed. Accuracy: " << acc);
        }

        CHECK_CUDA(cudaDeviceSynchronize());
        printf("Epoch %d completed.\n", epoch);
    }

    saveWeights(save_weights_path);

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

    CHECK_CUDA(cudaFreeHost(h_loss_infer));
    CHECK_CUDA(cudaFree(d_loss_infer));
    CHECK_CUDA(cudaFreeHost(h_count_correct_infer));
    CHECK_CUDA(cudaFree(d_count_correct_infer));

    CHECK_CUDA(cudaFree(d_train_images));
    CHECK_CUDA(cudaFree(d_train_labels));
}

void infer() {

}

int main(int argc, char* argv[]) {
    parseArguments(argc, argv);
    LOG("parsed arguments");

    if (!infer_mode) {
        train();
    } else {
        infer();
    }

    return 0;
}