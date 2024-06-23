#include "add.cu"

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