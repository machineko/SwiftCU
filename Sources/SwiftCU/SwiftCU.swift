import cxxCu
import Foundation

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
/// custom inits
extension CUDevice {}

extension cudaError: Equatable {
    static func == (lhs: cudaError, rhs: cudaError) -> Bool {
        return lhs.rawValue == rhs.rawValue
    }
}

extension cudaError {
    var isSuccessful: Bool {
        return self == .cudaSuccess
    }
}

extension cudaError_t {
    var asSwift: cudaError {
        return cudaError(rawValue: Int(self.rawValue))!
    }
}

/// Device creation extensions
extension CUDevice {
    func setDevice() -> Bool {
        let status = cudaSetDevice(self.index).asSwift
        #if safetyCheck
            precondition(status.isSuccessful, "Can't create device at idx: \(self.index) cudaErrorValue: \(status)")
        #endif
        return status.isSuccessful
    }
}


extension cudaUUID_t {
     var asSwift: UUID {
        let uuidBytes = withUnsafeBytes(of: self.bytes) { Data($0) }
        return UUID(uuid: uuidBytes.withUnsafeBytes { $0.load(as: uuid_t.self) })
    }
}

extension cudaDeviceProp {
    var deviceName: String {
        return withUnsafeBytes(of: self.name) { $0.bindMemory(to: CChar.self).baseAddress.map { String(cString: $0) } ?? "" }
    }
}


extension CUDevice {
    func getDeviceProperties() -> cudaDeviceProp {
        var properties = cudaDeviceProp.init()
        let status = cudaGetDeviceProperties_v2(&properties, self.index).asSwift
        if status.isSuccessful {
            return properties
        }
        else {
            fatalError("Failed to get device properties for idx: \(self.index) cudaErrodrValue: \(status)")
        }
    }

}