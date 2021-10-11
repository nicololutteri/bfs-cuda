#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CHECK_CUDA_ERROR                                                       \
    {                                                                          \
        cudaDeviceSynchronize();                                               \
        cuda_error::getLastCudaError(__FILE__, __LINE__, __func__);            \
    }

#define SAFE_CALL(function)                                                    \
    {                                                                          \
        cuda_error::safe_call(function, __FILE__, __LINE__, __func__);         \
    }

//------------------------------------------------------------------------------

namespace cuda_error {

static void getLastCudaError(const char* file, int line, const char* func_name);

static void safe_call(cudaError_t error,
               const char* file,
               int         line,
               const char* func_name);

static void cudaErrorHandler(cudaError_t error,
                      const char* error_message,
                      const char* file,
                      int         line,
                      const char* func_name);

} // namespace cuda_error

#include <cuda_runtime.h>    // cudaError_t
#include <cassert>
#include <iostream>

namespace cuda_error {

void getLastCudaError(const char* file, int line, const char* func_name) {
    cudaErrorHandler(cudaGetLastError(), "", file, line, func_name);
}

void safe_call(cudaError_t error,
               const char* file,
               int         line,
               const char* func_name) {
    cudaErrorHandler(error, "", file, line, func_name);
}

void cudaErrorHandler(cudaError_t error,
                      const char* error_message,
                      const char* file,
                      int         line,
                      const char* func_name) {
    if (cudaSuccess != error) {
        std::cerr << "\nCUDA error\n" << file << "(" << line << ")"
                  << " [ " << func_name << " ] : " << error_message
                  << " -> " << cudaGetErrorString(error)
                  << "(" << static_cast<int>(error) << ")\n"
                  << std::endl;
        assert(false);                                                  //NOLINT
        std::atexit(reinterpret_cast<void(*)()>(cudaDeviceReset));
        std::exit(EXIT_FAILURE);
    }
}

} // namespace cuda_error

