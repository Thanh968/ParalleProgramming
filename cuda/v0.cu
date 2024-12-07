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
            dim3 grid_size((m + block_size.y - 1) / block_size.y, (k + block_size.x - 1) / block_size.x);
            g_matMul<<<grid_size, block_size>>>(mat_a.getData(), mat_b.getData(), mat_out.getData(), m, n, k);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        static void reLU(const Matrix& mat_in, Matrix& mat_out) {
            int m = mat_in.getHeight(), n = mat_in.getWidth();
            dim3 block_size(DEFAULT_TILEWIDTH, DEFAULT_TILEWIDTH);
            dim3 grid_size((m + block_size.y - 1) / block_size.y, (n + block_size.x - 1) / block_size.x);
            g_matReLU<<<grid_size, block_size>>>(mat_in.getData(), mat_out.getData(), m, n);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        static void softmax(const Matrix& mat_in, Matrix& mat_out) {
            int m = mat_in.getHeight(), n = mat_in.getWidth();
            float* h_data = new float[m * n];
            float sum;

            CHECK_CUDA(cudaMemcpy(h_data, mat_in.getData(), m * n * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < m; ++i) {
                sum = 0;
                for (int j = 0; j < n; ++j) {
                    h_data[i * n + j] = exp(h_data[i * n + j]);
                    sum += h_data[i * n + j];
                }
                for (int j = 0; j < n; ++j) {
                    h_data[i * n + j] /= sum;
                }
            }
            CHECK_CUDA(cudaMemcpy(mat_out.getData(), h_data, m * n * sizeof(float), cudaMemcpyHostToDevice));
            delete[] h_data;
        }
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
}

namespace MathOp {
    using namespace Data;

    class DifferentiableScalarFunction {
    public:
        virtual void output(const Matrix& mat_in, Matrix& mat_out) = 0;
    };

    class ReLU : public DifferentiableScalarFunction {
    public:
        void output(const Matrix& mat_in, Matrix& mat_out) override {
            Matrix::reLU(mat_in, mat_out);
        }
    };

    class Softmax : public DifferentiableScalarFunction {
    public:
        void output(const Matrix& mat_in, Matrix& mat_out) override {
            Matrix::softmax(mat_in, mat_out);
        }
    };

    enum class ActivFuncEnum {
        RELU,
        SOFTMAX,
    };
}

namespace Module {
    using namespace Data;
    using namespace MathOp;

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
        Matrix* w_;
        Vector* b_;
        Matrix* m_;
        Matrix* z_;
        Matrix* a_;
        Matrix* grad_w_;
        Matrix* grad_b_;
        Matrix* grad_;
        DifferentiableScalarFunction* activ_func_;
    public:
        Dense(int input_size, int output_size, ActivFuncEnum activ_func) :
            input_size_(input_size),
            output_size_(output_size),
            w_(new Matrix(input_size, output_size)),
            b_(new Vector(output_size)),
            m_(nullptr),
            z_(nullptr),
            a_(nullptr),
            grad_w_(nullptr),
            grad_b_(nullptr),
            grad_(nullptr),
            activ_func_(nullptr)
            {
                if (activ_func == ActivFuncEnum::RELU) {
                    LOG("activ_func set to ReLU");
                    activ_func_ = new ReLU();
                }
                else if (activ_func == ActivFuncEnum::SOFTMAX) {
                    LOG("activ_func set to Softmax");
                    activ_func_ = new Softmax();
                }
            }

        ~Dense() {
            LOG("Destroy dense layer...");
            delete w_;
            delete b_;
            delete m_;
            delete z_;
            delete a_;
            delete activ_func_;
        }
    public:
        void init(int batch_size = 1, bool randomize_weight = false) override {
            if (m_) delete m_;
            if (z_) delete z_;
            if (a_) delete a_;

            AbstractModule::init(batch_size, randomize_weight);
            m_ = new Matrix(batch_size_, output_size_);
            z_ = new Matrix(batch_size_, output_size_);
            a_ = new Matrix(batch_size_, output_size_);

            if (randomize_weight) {
                w_->randomizeValues();
                b_->randomizeValues();
            }

            #ifdef DEBUG
            printf("Weight:\n");
            w_->print();
            printf("Bias:\n");
            b_->print();
            #endif
        }

        Matrix* getOutput() const override {
            return a_;
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
            activ_func_->output(*z_, *a_);
            #ifdef DEBUG
            printf("a = f(z)\n");
            a_->print();
            #endif
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