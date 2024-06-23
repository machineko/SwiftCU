import cxxCu
import Foundation
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

extension cudaMemoryCopyType {
    var asCUDA: cudaMemcpyKind {
        switch self {
            case .cudaMemcpyHostToHost:
                return cudaMemcpyKind.init(0)
            case .cudaMemcpyHostToDevice:
                return cudaMemcpyKind.init(1)
            case .cudaMemcpyDeviceToHost:
                return cudaMemcpyKind.init(2)
            case .cudaMemcpyDeviceToDevice:
                return cudaMemcpyKind.init(3)
            case .cudaMemcpyDefault:
                return cudaMemcpyKind.init(4)
        }
    }
}

extension UnsafeMutableRawPointer? {
    // totalSize -> size in bytes
    mutating func cudaMemoryAllocate(_ totalSize: Int) -> cudaError {
        var tempPointer: UnsafeMutableRawPointer? = self
        print("Before cudaMalloc: tempPointer = \(String(describing: tempPointer))")
        let status = withUnsafeMutablePointer(to: &tempPointer) { pointer in
            cudaMalloc(pointer, totalSize).asSwift
        }
        print("After cudaMalloc: tempPointer = \(String(describing: tempPointer))")
        self = tempPointer
        print("After assignment: self = \(String(describing: self))")
        #if safetyCheck
        precondition(status.isSuccessful, "Can't allocate memory on device cudaErrorValue: \(status)")
        #endif
        return status
    }

    mutating func cudaMemoryCopy(to: [Float32], numberOfBytes: size_t, copyKind: cudaMemoryCopyType) -> cudaError {
        let status = cudaMemcpy(self, to, numberOfBytes, copyKind.asCUDA).asSwift

         #if safetyCheck
            precondition(status.isSuccessful, "Can't allocate memory on device cudaErrorValue: \(status)")
        #endif
        return status
    }
}

func cudaMemoryCopy(from: UnsafeMutableRawPointer?, to: UnsafeRawPointer?, numberOfBytes: size_t, copyKind: cudaMemoryCopyType) -> cudaError {
    let status = cudaMemcpy(from, to, numberOfBytes, cudaMemcpyHostToDevice).asSwift
    #if safetyCheck
        precondition(status.isSuccessful, "Can't allocate memory on device cudaErrorValue: \(status)")
    #endif
    return status
}