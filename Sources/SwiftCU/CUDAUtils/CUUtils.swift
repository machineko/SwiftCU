import Foundation
import cxxCU

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

    @inline(__always)
    func safetyCheckCondition(message: String) {
        precondition(self.isSuccessful, "\(message): cudaErrorValue: \(self)")
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
        } else {
            fatalError("Failed to get device properties for idx: \(self.index) cudaErrodrValue: \(status)")
        }
    }
}

/// Device creation extensions
extension CUDevice {
    func setDevice() -> Bool {
        let status = cudaSetDevice(self.index).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't create device at idx: \(self.index)")
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

struct cudaStream: ~Copyable {
    var stream: cudaStream_t?

    init() {
        let status = cudaStreamCreate(&self.stream).asSwift
        print(status, "XD")
        status.safetyCheckCondition(message: "Can't create stream")
    }

    func sync() -> cudaError {
        let status = cudaStreamSynchronize(self.stream).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't synchronize stream")
        #endif
        return status
    }

    deinit {
        let status = cudaStreamDestroy(self.stream)
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't destroy stream")
        #endif
    }
}

extension UnsafeMutableRawPointer? {
    mutating func cudaMemoryAllocate(_ totalSize: Int) -> cudaError {
        let status = cudaMalloc(&self, totalSize).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't allocate memory on device")
        #endif
        return status
    }

    mutating func cudaMemoryCopy(fromRawPointer: UnsafeRawPointer, numberOfBytes: size_t, copyKind: cudaMemoryCopyType) -> cudaError {
        let status = cudaMemcpy(self, fromRawPointer, numberOfBytes, copyKind.asCUDA).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't copy memory from UnsafeRawPointer copyKind \(copyKind)")
        #endif
        return status
    }

    mutating func cudaMemoryCopy(fromMutableRawPointer: UnsafeMutableRawPointer?, numberOfBytes: size_t, copyKind: cudaMemoryCopyType)
        -> cudaError
    {
        let status = cudaMemcpy(self, fromMutableRawPointer, numberOfBytes, copyKind.asCUDA).asSwift

        #if safetyCheck
            status.safetyCheckCondition(message: "Can't copy memory from UnsafeRawPointer copyKind \(copyKind)")
        #endif
        return status
    }

    mutating func cudaMemoryCopyAsync(
        fromRawPointer: UnsafeRawPointer, numberOfBytes: size_t, copyKind: cudaMemoryCopyType, stream: borrowing cudaStream
    ) -> cudaError {
        let status = cudaMemcpyAsync(self, fromRawPointer, numberOfBytes, copyKind.asCUDA).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't copy memory from UnsafeRawPointer copyKind \(copyKind)")
        #endif
        return status
    }

    mutating func cudaMemoryCopyAsync(
        fromMutableRawPointer: UnsafeMutableRawPointer?, numberOfBytes: size_t, copyKind: cudaMemoryCopyType, stream: borrowing cudaStream
    ) -> cudaError {
        let status = cudaMemcpyAsync(self, fromMutableRawPointer, numberOfBytes, copyKind.asCUDA, stream.stream).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't copy memory from UnsafeRawPointer copyKind \(copyKind)")
        #endif
        return status
    }

}
