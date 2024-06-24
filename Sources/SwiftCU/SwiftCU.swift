import cxxCU

extension UnsafeMutableRawPointer? {
    var cuPoint: UnsafeMutablePointer<UnsafeMutableRawPointer?> {
        let argumentPointer = UnsafeMutablePointer<UnsafeMutableRawPointer?>.allocate(capacity: 1)
        argumentPointer.initialize(to: self)
        return argumentPointer
    }
}

// Pointer to pointer magic for cuda calling
public struct CUDAKernelArguments: ~Copyable {

    var argumentPointers: [UnsafeMutablePointer<UnsafeMutableRawPointer?>] = []
    var allocatedArguments: UnsafeMutablePointer<UnsafeMutableRawPointer?>?

    // Add normal pointer to classical type (Int/Float etc.)
    mutating func addPointer(_ pointer: UnsafeMutablePointer<some Any>?) {
        let rawPointer = UnsafeMutableRawPointer(pointer)
        let argumentPointer = UnsafeMutablePointer<UnsafeMutableRawPointer?>.allocate(capacity: 1)
        argumentPointer.initialize(to: rawPointer)
        argumentPointers.append(argumentPointer)
    }

    // Pointer to pointer for single UnsafeMutableRawPointer
    mutating func addRawPointer(_ pointer: UnsafeMutableRawPointer?) {
        let argumentPointer = UnsafeMutablePointer<UnsafeMutableRawPointer?>.allocate(capacity: 1)
        argumentPointer.initialize(to: pointer)
        argumentPointers.append(argumentPointer)
    }

    mutating func addRawPointers(_ pointers: [UnsafeMutableRawPointer?]) {
        for p in pointers {
            let argumentPointer = UnsafeMutablePointer<UnsafeMutableRawPointer?>.allocate(capacity: 1)
            argumentPointer.initialize(to: p.cuPoint)
            argumentPointers.append(argumentPointer)
        }
    }

    mutating func getArgsPointer() -> UnsafeMutablePointer<UnsafeMutableRawPointer?>? {
        let count = argumentPointers.count
        allocatedArguments = UnsafeMutablePointer<UnsafeMutableRawPointer?>.allocate(capacity: count)
        allocatedArguments?.initialize(from: argumentPointers.map(\.pointee), count: count)
        return allocatedArguments
    }

    deinit {
        allocatedArguments?.deallocate()
        for argumentPointer in argumentPointers {
            argumentPointer.deinitialize(count: 1)
            argumentPointer.deallocate()
        }
    }
}

public struct CUDevice: Sendable {
    var index: Int32 = 0
}


public struct CUDAKernel {
    let functionPointer: UnsafeRawPointer?

    func launch(arguments: consuming CUDAKernelArguments, blockDim: dim3, gridDim: dim3, sharedMemory: Int = 0) -> cudaError {
        let status = cudaLaunchKernel(
            self.functionPointer, gridDim, blockDim, arguments.getArgsPointer(), sharedMemory, nil
        ).asSwift
        #if safetyCheck
            precondition(status.isSuccessful, "Can't launch kernel cudaErrorValue: \(status)")
        #endif
        return status
    }

}