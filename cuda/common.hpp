#pragma once

#include <iostream>

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
