#include <cuda_runtime.h>

#ifdef rtx3090Test 
enum KernelID {
    ADD_F32,
};
void* getKernelPointer(KernelID id);
#else
#endif