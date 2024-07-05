import Foundation
import cxxCU

/// Custom initializers for `CUDevice`
extension CUDevice {}

/// Conformance to `Equatable` for `cudaError`
extension cudaError: Equatable {
    /// Checks if two `cudaError` values are equal
    /// - Parameters:
    ///   - lhs: The left-hand side `cudaError` value.
    ///   - rhs: The right-hand side `cudaError` value.
    /// - Returns: A Boolean value indicating whether the two values are equal.
    static func == (lhs: cudaError, rhs: cudaError) -> Bool {
        return lhs.rawValue == rhs.rawValue
    }
}

extension cudaError {
    /// Checks if the error represents a successful CUDA operation.
    var isSuccessful: Bool {
        return self == .cudaSuccess
    }

    /// Checks the condition and throws a precondition failure if the error is not successful.
    /// - Parameter message: The message to include in the precondition failure.
    @inline(__always)
    func safetyCheckCondition(message: String) {
        precondition(self.isSuccessful, "\(message): cudaErrorValue: \(self)")
    }
}

extension cudaError_t {
    /// Converts the `cudaError_t` to a Swift `cudaError`.
    var asSwift: cudaError {
        return cudaError(rawValue: Int(self.rawValue))!
    }
}

extension cudaUUID_t {
    /// Converts the `cudaUUID_t` to a Swift `UUID`.
    var asSwift: UUID {
        let uuidBytes = withUnsafeBytes(of: self.bytes) { Data($0) }
        return UUID(uuid: uuidBytes.withUnsafeBytes { $0.load(as: uuid_t.self) })
    }
}

extension cudaDeviceProp {
    /// Retrieves the device name as a `String`.
    var deviceName: String {
        return withUnsafeBytes(of: self.name) { $0.bindMemory(to: CChar.self).baseAddress.map { String(cString: $0) } ?? "" }
    }
}

extension CUDevice {
    /// Retrieves the properties of the CUDA device.
    /// - Returns: The `cudaDeviceProp` structure containing the device properties.
    /// - Note: This function will terminate the program if it fails to get the device properties.
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

/// Device creation extensions for `CUDevice`
extension CUDevice {
    /// Sets the current CUDA device.
    /// - Returns: A Boolean value indicating whether the device was successfully set.
    func setDevice() -> Bool {
        let status = cudaSetDevice(self.index).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't create device at idx: \(self.index)")
        #endif
        return status.isSuccessful
    }
}

extension cudaMemoryCopyType {
    /// Converts the `cudaMemoryCopyType` to a CUDA `cudaMemcpyKind`.
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

/// A structure representing a CUDA stream.
struct cudaStream: ~Copyable {
    var stream: cudaStream_t?

    /// Initializes a new CUDA stream.
    init() {
        let status = cudaStreamCreate(&self.stream).asSwift
        status.safetyCheckCondition(message: "Can't create stream")
    }

    /// Synchronizes the CUDA stream.
    /// - Returns: The `cudaError` indicating the result of the synchronization operation.
    func sync() -> cudaError {
        let status = cudaStreamSynchronize(self.stream).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't synchronize stream")
        #endif
        return status
    }

    /// Destroys the CUDA stream.
    deinit {
        let status = cudaStreamDestroy(self.stream)
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't destroy stream")
        #endif
    }
}

extension UnsafeMutableRawPointer? {
    /// Checks if the pointer is allocated on the device.
    /// - Returns: A Boolean value indicating whether the pointer is allocated on the device.
    func isAllocatedOnDevice() -> Bool {
        return self.getCUDAAttributes().devicePointer != nil
    }

    /// Checks if the pointer is allocated on the host.
    /// - Returns: A Boolean value indicating whether the pointer is allocated on the host.
    func isAllocatedOnHost() -> Bool {
        return self.getCUDAAttributes().hostPointer != nil
    }

    /// Retrieves the CUDA attributes of the pointer.
    /// - Returns: The `cudaPointerAttributes` structure containing the pointer attributes.
    func getCUDAAttributes() -> cudaPointerAttributes {
        var attributes: cudaPointerAttributes = cudaPointerAttributes()
        let status = cudaPointerGetAttributes(&attributes, self).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't get cuda pointer attributes")
        #endif
        return attributes
    }
}

extension UnsafeMutableRawPointer? {
    /// Deallocates device memory.
    /// - Returns: The `cudaError` indicating the result of the deallocation operation.
    mutating func cudaDeallocate() -> cudaError {
        let status = cudaFree(self).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't free memory on device")
        #endif
        return status
    }

    /// Deallocates memory on the device and sets the pointer to nil to ensure proper cleanup of pointers. This won't deallocate host data.
    /// - Returns: The `cudaError` indicating the result of the deallocation operation.
    mutating func cudaAndHostDeallocate() -> cudaError {
        let status = cudaFree(self).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't free memory on device")
        #endif
        self = nil
        return status
    }

    /// Allocates memory on the device.
    /// - Parameter totalSize: The total size of memory to allocate.
    /// - Returns: The `cudaError` indicating the result of the allocation operation.
    mutating func cudaMemoryAllocate(_ totalSize: Int) -> cudaError {
        let status = cudaMalloc(&self, totalSize).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't allocate memory on device")
        #endif
        return status
    }

    /// Copies memory from a raw pointer to the device.
    /// - Parameters:
    ///   - fromRawPointer: The source raw pointer.
    ///   - numberOfBytes: The number of bytes to copy.
    ///   - copyKind: The type of memory copy operation.
    /// - Returns: The `cudaError` indicating the result of the copy operation.
    mutating func cudaMemoryCopy(fromRawPointer: UnsafeRawPointer, numberOfBytes: size_t, copyKind: cudaMemoryCopyType) -> cudaError {
        let status = cudaMemcpy(self, fromRawPointer, numberOfBytes, copyKind.asCUDA).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't copy memory from UnsafeRawPointer copyKind \(copyKind)")
        #endif
        return status
    }

    /// Copies memory from a mutable raw pointer to the device.
    /// - Parameters:
    ///   - fromMutableRawPointer: The source mutable raw pointer.
    ///   - numberOfBytes: The number of bytes to copy.
    ///   - copyKind: The type of memory copy operation.
    /// - Returns: The `cudaError` indicating the result of the copy operation.
    mutating func cudaMemoryCopy(fromMutableRawPointer: UnsafeMutableRawPointer?, numberOfBytes: size_t, copyKind: cudaMemoryCopyType)
        -> cudaError
    {
        let status = cudaMemcpy(self, fromMutableRawPointer, numberOfBytes, copyKind.asCUDA).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't copy memory from UnsafeRawPointer copyKind \(copyKind)")
        #endif
        return status
    }

    /// Copies memory asynchronously from a raw pointer to the device.
    /// - Parameters:
    ///   - fromRawPointer: The source raw pointer.
    ///   - numberOfBytes: The number of bytes to copy.
    ///   - copyKind: The type of memory copy operation.
    ///   - stream: The CUDA stream to use for the asynchronous copy.
    /// - Returns: The `cudaError` indicating the result of the copy operation.
    mutating func cudaMemoryCopyAsync(
        fromRawPointer: UnsafeRawPointer, numberOfBytes: size_t, copyKind: cudaMemoryCopyType, stream: borrowing cudaStream
    ) -> cudaError {
        let status = cudaMemcpyAsync(self, fromRawPointer, numberOfBytes, copyKind.asCUDA).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't copy memory from UnsafeRawPointer copyKind \(copyKind)")
        #endif
        return status
    }

    /// Copies memory asynchronously from a mutable raw pointer to the device.
    /// - Parameters:
    ///   - fromMutableRawPointer: The source mutable raw pointer.
    ///   - numberOfBytes: The number of bytes to copy.
    ///   - copyKind: The type of memory copy operation.
    ///   - stream: The CUDA stream to use for the asynchronous copy.
    /// - Returns: The `cudaError` indicating the result of the copy operation.
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

extension CUMemory {
    /// Updates the CUDA memory information.
    /// - Returns: The `cudaError` indicating the result of the update operation.
    mutating func updateCUDAMemory() -> cudaError {
        let status = cudaMemGetInfo(&self.free, &self.total).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't copy memory from UnsafeRawPointer copyKind \(copyKind)")
        #endif
        return status
    }
}
