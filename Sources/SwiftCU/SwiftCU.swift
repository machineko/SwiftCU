import cxxCu


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

struct CUDevice {
    var index: Int32 = 0
}

extension cudaError {
    var isSuccessful: Bool {
        return self == .init(0)
    }
}

extension CUDevice {
    func setDevice() -> Bool {
        let status: cudaError = cudaSetDevice(self.index)
        return status.isSuccessful
    }
}