import cxxCU
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

func cudaMemoryAllocate(pointer: inout UnsafeMutableRawPointer?, _ totalSize: Int) -> cudaError {    
    let status = cudaMalloc(&pointer, totalSize).asSwift
    

    #if safetyCheck
        precondition(status.isSuccessful, "Can't allocate memory on device cudaErrorValue: \(status)")
    #endif

    return status
}
extension UnsafeMutableRawPointer? {

     mutating func cudaMemoryAllocate(_ totalSize: Int) -> cudaError {
        let status = cudaMalloc(&self, totalSize).asSwift

        #if safetyCheck
            precondition(status.isSuccessful, "Can't allocate memory on device cudaErrorValue: \(status)")
        #endif

        return status
    }

    mutating func cudaMemoryCopy(fromRawPointer: UnsafeRawPointer, numberOfBytes: size_t, copyKind: cudaMemoryCopyType) -> cudaError {
        let status = cudaMemcpy(self, fromRawPointer, numberOfBytes, copyKind.asCUDA).asSwift

         #if safetyCheck
            precondition(status.isSuccessful, "Can't allocate memory on device cudaErrorValue: \(status)")
        #endif
        return status
    }

    mutating func cudaMemoryCopy(fromMutableRawPointer: UnsafeMutableRawPointer?, numberOfBytes: size_t, copyKind: cudaMemoryCopyType) -> cudaError {
        let status = cudaMemcpy(self, fromMutableRawPointer, numberOfBytes, copyKind.asCUDA).asSwift

         #if safetyCheck
            precondition(status.isSuccessful, "Can't allocate memory on device cudaErrorValue: \(status)")
        #endif
        return status
    }
}
