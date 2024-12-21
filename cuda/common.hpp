#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
using namespace std;

#define DEFAULT_BLOCKSIZE 256
#define DEFAULT_TILEWIDTH 32
#define BM 128
#define BN 8
#define BK 128
#define TM 8
#define TK 8

#define CHECK_CUDA(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}
#define LOG(...) \
{ \
    std::cout << __VA_ARGS__ << std::endl; \
}

#ifdef DEBUG
#define BREAK CHECK_CUDA(cudaDeviceSynchronize());
#else
#define BREAK {}
#endif

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


class DataFrame {
private:
    vector<string> column_names;
    vector<vector<float>> data;

public:
    DataFrame(const vector<string>& col_names) : column_names(col_names) {}

    void addRow(const vector<float>& row) {
        if (row.size() != column_names.size()) {
            throw invalid_argument("Row size must match the number of columns");
        }
        data.push_back(row);
    }

    void print() const {
        for (const auto& name : column_names) {
            cout << std::setw(10) << name << " ";
        }
        cout << "\n";

        for (const auto& row : data) {
            for (const auto& value : row) {
                cout << std::setw(10) << value << " ";
            }
            cout << "\n";
        }
    }
    void saveToFile(const string& filename) const {
        ofstream outfile(filename);
        if (!outfile) {
            throw runtime_error("Unable to open file: " + filename);
        }

        for (const auto& name : column_names) {
            outfile << name << ",";
        }
        outfile << "\n";

        for (const auto& row : data) {
            for (const auto& value : row) {
                outfile << value << ",";
            }
            outfile << "\n";
        }

        outfile.close();
    }

    
};