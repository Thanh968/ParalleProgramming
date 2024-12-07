#pragma once

#ifdef DEBUG
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
#else
#define CHECK_CUDA(call) call
#define LOG(...) {}
#endif

#define DEFAULT_BLOCKSIZE 256
#define DEFAULT_TILEWIDTH 32