#include <cuda_runtime.h>

__global__ void addKernel(const float *a, const float *b, float *c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

enum KernelID {
    ADD_F32,
};

void* getKernelPointer(KernelID id) {
    switch (id) {
        case ADD_F32:
            return (void*)addKernel;
        default:
            return nullptr;
    }
}