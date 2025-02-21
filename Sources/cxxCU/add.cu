#include <cuda_fp16.h>
__global__ void addKernel(const float *a, const float *b, float *c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

__global__ void addF16(const half *a, const half *b, half *c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

enum KernelID {
    ADD_F32,
    ADD_F16,
};

void* getKernelPointer(KernelID id) {
    switch (id) {
        case ADD_F32:
            return (void*)addKernel;
        case ADD_F16:
            return (void*)addF16;
        default:
            return nullptr;
    }
}