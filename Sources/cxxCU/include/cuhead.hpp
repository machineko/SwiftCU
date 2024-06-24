#include <cuda_runtime.h>

enum KernelID {
    ADD_F32,
};

void* getKernelPointer(KernelID id);