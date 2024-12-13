#include "common.hpp"

#include <iostream>
#include <memory>
#include <vector>
#include <curand_kernel.h>

using namespace std;

__global__ void g_randomizeValues(float* a, int n, unsigned long long seed = 666) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        a[idx] = curand_uniform(&state) * 2.0f - 1.0f;
    }
}

__global__ void g_matMul(float* mat_a, float* mat_b, float* mat_out, int m, int n, int k) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    float out_rc = 0;
    if (r < m && k < n) {
        for (int i = 0; i < n; ++i) {
            out_rc += mat_a[r * n + i] * mat_b[i * k + c];
        }
        mat_out[r * k + c] = out_rc;
    }
}

__global__ void g_matAddRows(float* mat_a, float* vec_b, float* mat_out, int m, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < m && c < n) {
        mat_out[r * n + c] = mat_a[r * n + c] + vec_b[c];
    }
}

__global__ void g_matReLU(float* mat_in, float* mat_out, int m, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < m && c < n) {
        mat_out[r * n + c] = max(mat_in[r * n + c], 0.0f);
    }
}

__global__ void g_matExpSumRows(float* mat_in, float* mat_out, float* sums, int m, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    if (r >= m || c >= n) return;
    
    mat_out[r * n + c] = exp(mat_in[r * n + c]);
    __syncthreads();

    for (int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride && c + stride < n) {
            mat_out[r * n + c] += mat_out[r * n + c + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(sums + r, mat_out[r * n + blockIdx.x * blockDim.x * 2]);
    }
}

__global__ void g_matSoftmax(float* mat_in, float* mat_out, float* sums, int m, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    if (r >= m || c >= n) return;
    
    mat_out[r * n + c] = exp(mat_in[r * n + c]) / sums[r];
}

namespace Data {
    class DeviceMemory {
    private:
        float* data_;
        int size_;

    public:
        DeviceMemory(int size) {
            LOG("+++ alloc +++")
            size_ = size;
            CHECK_CUDA(cudaMalloc((void**)&data_, size_ * sizeof(float)));
        }

        ~DeviceMemory() {
            LOG("--- dealloc ---")
            CHECK_CUDA(cudaFree(data_));
        }

    public:
        int getSize() const { return size_; }
        float* getData() const { return data_; }
        void setData(initializer_list<float> data) {
            if (data.size() > getSize()) return;
            vector<float> h_data(data);
            CHECK_CUDA(cudaMemcpy(data_, h_data.data(), h_data.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
    };

    class Matrix {
    private:
        shared_ptr<DeviceMemory> data_;
        int height_;
        int width_;
    
    public:
        Matrix(int height, int width) : height_(height), width_(width), data_(make_shared<DeviceMemory>(height * width)) {}
    
    public:
        int getHeight() const { return height_; }
        int getWidth() const { return width_; }
        float* getData() const { return data_->getData(); }
        void setData(initializer_list<float> data) {
            data_->setData(data);
        }
        void randomizeValues() {
            dim3 block_size(DEFAULT_BLOCKSIZE);
            dim3 grid_size((data_->getSize() + block_size.x - 1) / block_size.x);
            g_randomizeValues<<<grid_size, block_size>>>(data_->getData(), data_->getSize());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        void print() const {
            float* h_data = new float[height_ * width_];
            CHECK_CUDA(cudaMemcpy(h_data, data_->getData(), data_->getSize() * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < height_; ++i) {
                for (int j = 0; j < width_; ++j) {
                    printf("%8.4f", h_data[i * width_ + j]);
                }
                printf("\n");
            }
            delete[] h_data;
        }

    public:
        static void mul(const Matrix& mat_a, const Matrix& mat_b, Matrix& mat_out) {
            int m = mat_a.getHeight(), n = mat_a.getWidth(), k = mat_out.getWidth();
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((k + block_size.x - 1) / block_size.x, (m + block_size.y - 1) / block_size.y);
            g_matMul<<<grid_size, block_size>>>(mat_a.getData(), mat_b.getData(), mat_out.getData(), m, n, k);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // static void reLU(const Matrix& mat_in, Matrix& mat_out) {
        //     int m = mat_in.getHeight(), n = mat_in.getWidth();
        //     dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
        //     dim3 grid_size((n + block_size.x - 1) / block_size.x, (m + block_size.y - 1) / block_size.y);
        //     g_matReLU<<<grid_size, block_size>>>(mat_in.getData(), mat_out.getData(), m, n);
        //     CHECK_CUDA(cudaDeviceSynchronize());
        // }

        // static void softmax(const Matrix& mat_in, Matrix& mat_out) {
        //     int m = mat_in.getHeight(), n = mat_in.getWidth();
        //     float* h_data = new float[m * n];
        //     float sum;

        //     CHECK_CUDA(cudaMemcpy(h_data, mat_in.getData(), m * n * sizeof(float), cudaMemcpyDeviceToHost));
        //     for (int i = 0; i < m; ++i) {
        //         sum = 0;
        //         for (int j = 0; j < n; ++j) {
        //             h_data[i * n + j] = exp(h_data[i * n + j]);
        //             sum += h_data[i * n + j];
        //         }
        //         for (int j = 0; j < n; ++j) {
        //             h_data[i * n + j] /= sum;
        //         }
        //     }
        //     CHECK_CUDA(cudaMemcpy(mat_out.getData(), h_data, m * n * sizeof(float), cudaMemcpyHostToDevice));
        //     delete[] h_data;
        // }

        // static void softmax(const Matrix& mat_in, Matrix& mat_out, Vector& sums) {
        //     int m = mat_in.getHeight(), n = mat_out.getWidth();
        //     dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
        //     dim3 grid_size((n + block_size.x - 1) / block_size.x, (m + block_size.y - 1) / block_size.y);
        //     g_matExpSumRows<<<grid_size, block_size>>>(mat_in.getData(), mat_out.getData(), sums.getData(), m, n);
        //     g_matSoftmax<<<grid_size, block_size>>>(mat_in.getData(), mat_out.getData(), sums.getData(), m, n);
        //     CHECK_CUDA(cudaDeviceSynchronize());
        // }
    };

    class Vector : public Matrix {
    public:
        Vector(int size) : Matrix(1, size) {}

    public:
        static void addRows(const Matrix& mat_a, const Vector& vec_b, Matrix& mat_out) {
            int m = mat_a.getHeight(), n = mat_a.getWidth();
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((m + block_size.y - 1) / block_size.y, (n + block_size.x - 1) / block_size.x);
            g_matAddRows<<<grid_size, block_size>>>(mat_a.getData(), vec_b.getData(), mat_out.getData(), m, n);
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    };

    void reLU(const Matrix& mat_in, Matrix& mat_out) {
        int m = mat_in.getHeight(), n = mat_in.getWidth();
        dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
        dim3 grid_size((n + block_size.x - 1) / block_size.x, (m + block_size.y - 1) / block_size.y);
        g_matReLU<<<grid_size, block_size>>>(mat_in.getData(), mat_out.getData(), m, n);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void softmax(const Matrix& mat_in, Matrix& mat_out, Vector& sums) {
        int m = mat_in.getHeight(), n = mat_out.getWidth();
        dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
        dim3 grid_size((n + block_size.x - 1) / block_size.x, (m + block_size.y - 1) / block_size.y);
        g_matExpSumRows<<<grid_size, block_size>>>(mat_in.getData(), mat_out.getData(), sums.getData(), m, n);
        CHECK_CUDA(cudaDeviceSynchronize());
        g_matSoftmax<<<grid_size, block_size>>>(mat_in.getData(), mat_out.getData(), sums.getData(), m, n);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}

namespace Module {
    using namespace Data;

    class AbstractModule {
    protected:
        int batch_size_;
    public:
        virtual void init(int batch_size = 1, bool randomize_weight = false) {
            batch_size_ = batch_size;
        }
        virtual Matrix* getOutput() const = 0;
        virtual Matrix* forward(Matrix* x) = 0;
    };

    class Linear : public AbstractModule {
    private:
        int input_size_;
        int output_size_;
        Matrix* w_;
        Vector* b_;
        Matrix* m_;
        Matrix* z_;

    public:
        Linear(int input_size, int output_size) :
            input_size_(input_size_),
            output_size_(output_size),
            w_(new Matrix(input_size, output_size)),
            b_(new Vector(output_size)),
            m_(nullptr),
            z_(nullptr)
            {}
        
        ~Linear() {
            LOG("Destroy Linear layer");
            delete w_;
            delete b_;
            delete m_;
            delete z_;
        }

    public:
        void init(int batch_size = 1, bool randomize_weight = false) override {
            if (m_ != nullptr) delete m_;
            if (z_ != nullptr) delete z_;

            AbstractModule::init(batch_size, randomize_weight);
            m_ = new Matrix(batch_size_, output_size_);
            z_ = new Matrix(batch_size_, output_size_);

            if (randomize_weight) {
                w_->randomizeValues();
                b_->randomizeValues();
            }
        }

        Matrix* getOutput() const override {
            return z_;
        }

        Matrix* forward(Matrix* x) override {
            Matrix::mul(*x, *w_, *m_);
            #ifdef DEBUG
            printf("m = x * W\n");
            m_->print();
            #endif
            Vector::addRows(*m_, *b_, *z_);
            #ifdef DEBUG
            printf("z = x * W + b\n");
            z_->print();
            #endif
            return getOutput();
        }
    };

    enum class ActivFuncEnum {
        RELU,
        SOFTMAX,
    };

    class ActivationLayer : public AbstractModule {
    protected:
        int size_;
        Matrix* a_;

    public:
        ActivationLayer(int size) : size_(size), a_(nullptr) {}

        ~ActivationLayer() {
            LOG("Destroy Activ Layer");
            delete a_;
        }

    public:
        void init(int batch_size = 1, bool randomize_weight = false) override {
            AbstractModule::init(batch_size, randomize_weight);
            if (a_) delete a_;
            a_ = new Matrix(batch_size_, size_);

            if (randomize_weight) {
                a_->randomizeValues();
            }
        }

        Matrix* getOutput() const override {
            return a_;
        }
    };

    class ReLU : public ActivationLayer {
    public:
        ReLU(int size) : ActivationLayer(size) {}

    public:
        Matrix* forward(Matrix* x) override {
            reLU(*x, *a_);
            return getOutput();
        }
    };

    class Softmax : public ActivationLayer {
    private:
        Vector* s_;

    public:
        Softmax(int size) : ActivationLayer(size), s_(nullptr) {}

        ~Softmax() {
            LOG("Destroy Softmax layer");
            delete s_;
        }

    public:
        void init(int batch_size = 1, bool randomize_weight = false) override {
            ActivationLayer::init(batch_size, randomize_weight);
            if (s_ != nullptr) delete s_;
            s_ = new Vector(size_);
        }

        Matrix* forward(Matrix* x) override {
            softmax(*x, *a_, *s_);
            return getOutput();
        }
    };

    struct DenseLayerConfig {
        int input_size_;
        int output_size_;
        ActivFuncEnum activ_func_;

        DenseLayerConfig(int input_size, int output_size, ActivFuncEnum activ_func) :
            input_size_(input_size),
            output_size_(output_size),
            activ_func_(activ_func)
            {}
    };

    class Dense : public AbstractModule {
    private:
        int input_size_;
        int output_size_;
        Linear* lin_;
        ActivationLayer* activ_;

    public:
        Dense(int input_size, int output_size, ActivFuncEnum activ_func) :
            input_size_(input_size),
            output_size_(output_size),
            lin_(new Linear(input_size, output_size)),
            activ_(nullptr) {
                if (activ_func == ActivFuncEnum::RELU) {
                    LOG("activ_ set to ReLU");
                    activ_ = new ReLU(output_size);
                } else if (activ_func == ActivFuncEnum::SOFTMAX) {
                    LOG("activ_ set to Softmax");
                    activ_ = new Softmax(output_size);
                }
            }

        ~Dense() {
            LOG("Destroy dense layer...");
            delete lin_;
            delete activ_;
        }

    public:
        void init(int batch_size = 1, bool randomize_weight = false) override {
            AbstractModule::init(batch_size, randomize_weight);
            lin_->init(batch_size, randomize_weight);
            activ_->init(batch_size, randomize_weight);
        }

        Matrix* getOutput() const override {
            return activ_->getOutput();
        }

        Matrix* forward(Matrix* x) override {
            lin_->forward(x);
            activ_->forward(lin_->getOutput());
            return getOutput();
        }
    };

    class Ann : public AbstractModule {
    private:
        vector<Dense*> dense_layers_;
    public:
        Ann(initializer_list<DenseLayerConfig> dense_layers_config) {
            dense_layers_ = vector<Dense*>();
            int n_dense_layers = dense_layers_config.size();
            auto it = dense_layers_config.begin();
            for (int i = 0; i < n_dense_layers; ++i, ++it) {
                auto config = *it;
                int input_size = config.input_size_;
                int output_size = config.output_size_;
                ActivFuncEnum activ_func = config.activ_func_;
                dense_layers_.emplace_back(new Dense(input_size, output_size, activ_func));
            }
        }

        ~Ann() {
            for (auto& dense_layer : dense_layers_) {
                delete dense_layer;
            }
            LOG("Destroy model...");
        }

    public:
        void init(int batch_size = 1, bool randomize_weight = false) override {
            AbstractModule::init(batch_size, randomize_weight);
            for (auto& layer : dense_layers_) {
                layer->init(batch_size, randomize_weight);
            }
        }

        Matrix* getOutput() const override {
            return dense_layers_[dense_layers_.size() - 1]->getOutput();
        }

        Matrix* forward(Matrix* x) override {
            printf("Input:\n");
            x->print();
            for (int i = 0; i < dense_layers_.size(); ++i) {
                LOG("Start layer " << i);
                x = dense_layers_[i]->forward(x);
                LOG("Layer " << i);
                #ifdef DEBUG
                x->print();
                #endif
            }
            return x;
        }
    };
}

using namespace Module;

int main() {
    Matrix* input = new Matrix({ 2, 4 });
    input->setData({ 1, 2, 3, 4, 5, 6, 7, 8 });
    input->print();

    Ann model({
        { 4, 3, ActivFuncEnum::RELU },
        { 3, 2, ActivFuncEnum::SOFTMAX }
    });
    model.init(2, true);

    Matrix* output = model.forward(input);
    printf("output:\n");
    output->print();
    delete input;

    return 0;
}