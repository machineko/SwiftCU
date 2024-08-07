import cxxCU

extension UnsafeMutableRawPointer? {
    public var cuPoint: UnsafeMutablePointer<UnsafeMutableRawPointer?> {
        let argumentPointer = UnsafeMutablePointer<UnsafeMutableRawPointer?>.allocate(capacity: 1)
        argumentPointer.initialize(to: self)
        return argumentPointer
    }
}

// Pointer to pointer magic for cuda calling
public struct CUDAKernelArguments: ~Copyable {

    public var argumentPointers: [UnsafeMutablePointer<UnsafeMutableRawPointer?>] = []
    public var allocatedArguments: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
    public var cuPointers: [UnsafeMutablePointer<UnsafeMutableRawPointer?>] = []

    public init() {
        self.argumentPointers = []
        self.cuPointers = []
    }

    // Add normal pointer to classical type (Int/Float etc.)
    public mutating func addPointer(_ pointer: UnsafeMutablePointer<some Any>?) {
        let rawPointer = UnsafeMutableRawPointer(pointer)
        let argumentPointer = UnsafeMutablePointer<UnsafeMutableRawPointer?>.allocate(capacity: 1)
        argumentPointer.initialize(to: rawPointer)
        argumentPointers.append(argumentPointer)
    }

    // Pointer to pointer for single UnsafeMutableRawPointer
    public mutating func addRawPointer(_ pointer: UnsafeMutableRawPointer?) {
        let argumentPointer = UnsafeMutablePointer<UnsafeMutableRawPointer?>.allocate(capacity: 1)
        argumentPointer.initialize(to: pointer)
        argumentPointers.append(argumentPointer)
    }

    public mutating func addRawPointers(_ pointers: [UnsafeMutableRawPointer?]) {
        for p in pointers {
            let argumentPointer = UnsafeMutablePointer<UnsafeMutableRawPointer?>.allocate(capacity: 1)
            let cuPoint = p.cuPoint
            argumentPointer.initialize(to: cuPoint)
            argumentPointers.append(argumentPointer)
            cuPointers.append(cuPoint)
        }
    }

    public mutating func getArgsPointer() -> UnsafeMutablePointer<UnsafeMutableRawPointer?>? {
        let count = argumentPointers.count
        allocatedArguments = UnsafeMutablePointer<UnsafeMutableRawPointer?>.allocate(capacity: count)
        allocatedArguments?.initialize(from: argumentPointers.map(\.pointee), count: count)
        return allocatedArguments
    }

    deinit {
        allocatedArguments?.deallocate()
        for (argPtr, cuPtr) in zip(argumentPointers, cuPointers) {
            argPtr.deinitialize(count: 1)
            cuPtr.deinitialize(count: 1)
            argPtr.deallocate()
            cuPtr.deallocate()
        }
    }
}

/// Represents a CUDA device.
public struct CUDADevice: Sendable, ~Copyable {
    public var index: Int32 = 0

    public init(index: Int32 = 0) {
        self.index = index
    }
}

/// Represents CUDA memory information.
public struct CUDAMemory: Sendable, ~Copyable {
    public var (free, total): (size_t, size_t)

    public init() {
        self.free = 0
        self.total = 0
    }
}

/// Represents a CUDA kernel.
public struct CUDAKernel: ~Copyable {
    public let functionPointer: UnsafeRawPointer?

    public init(functionPointer: UnsafeRawPointer?) {
        self.functionPointer = functionPointer
    }

    /// Launches the CUDA kernel with the specified arguments and configuration.
    /// - Parameters:
    ///   - arguments: The kernel arguments.
    ///   - blockDim: The block dimensions.
    ///   - gridDim: The grid dimensions.
    ///   - sharedMemory: The amount of shared memory to be used.
    /// - Returns: The CUDA error status.
    public func launch(arguments: consuming CUDAKernelArguments, blockDim: dim3, gridDim: dim3, sharedMemory: Int = 0) -> cudaError {
        let status = cudaLaunchKernel(
            self.functionPointer, gridDim, blockDim, arguments.getArgsPointer(), sharedMemory, nil
        ).asSwift
        #if safetyCheck
            precondition(status.isSuccessful, "Can't launch kernel cudaErrorValue: \(status)")
        #endif
        return status
    }

    /// Launches the CUDA kernel with the specified arguments and configuration, using a specific stream.
    /// - Parameters:
    ///   - arguments: The kernel arguments.
    ///   - blockDim: The block dimensions.
    ///   - gridDim: The grid dimensions.
    ///   - stream: The CUDA stream to be used.
    ///   - sharedMemory: The amount of shared memory to be used.
    /// - Returns: The CUDA error status.
    public func launch(
        arguments: consuming CUDAKernelArguments, blockDim: dim3, gridDim: dim3, stream: borrowing cudaStream, sharedMemory: Int = 0
    ) -> cudaError {
        let status = cudaLaunchKernel(
            self.functionPointer, gridDim, blockDim, arguments.getArgsPointer(), sharedMemory, stream.stream
        ).asSwift
        #if safetyCheck
            precondition(status.isSuccessful, "Can't launch kernel cudaErrorValue: \(status)")
        #endif
        return status
    }
}
