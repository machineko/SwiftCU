import Foundation
import Testing

@testable import SwiftCU
@testable import cxxCU

struct CUDAMemBasicTest {

    @Test func testCUDAAsyncCopy() async throws {
        let cuStatus = CUDevice(index: 0).setDevice()
        #expect(cuStatus)
        let stream = cudaStream()
        #expect(stream.stream != nil)
        var hostArray: [Float32] = [1.0, 2.0, 3.0]
        var outArray = [Float32](repeating: 0.0, count: hostArray.count)
        let arrayBytes: Int = hostArray.count * MemoryLayout<Float32>.size

        var aPointer: UnsafeMutableRawPointer?
        defer {
            let deallocateStatus = aPointer.cudaAndHostDeallocate()
            #expect(deallocateStatus.isSuccessful, "Can't dealocate aPointer \(deallocateStatus)")
        }
        let allocateStatus = aPointer.cudaMemoryAllocate(arrayBytes)
        #expect(allocateStatus.isSuccessful, "Can't allocate memory for aPointer \(allocateStatus)")

        var copyStatus = aPointer.cudaMemoryCopyAsync(
            fromRawPointer: &hostArray, numberOfBytes: arrayBytes, copyKind: .cudaMemcpyHostToDevice, stream: stream)
        #expect(copyStatus.isSuccessful, "Can't copy memory for aPointer \(copyStatus)")

        outArray.withUnsafeMutableBytes { rawBufferPointer in
            var address: UnsafeMutableRawPointer? = rawBufferPointer.baseAddress
            copyStatus = address.cudaMemoryCopyAsync(
                fromMutableRawPointer: aPointer, numberOfBytes: arrayBytes, copyKind: .cudaMemcpyDeviceToHost, stream: stream)
            #expect(copyStatus.isSuccessful, "Can't copy memory from device \(copyStatus)")
        }

        let syncStatus = stream.sync()
        #expect(syncStatus.isSuccessful, "Can't sync stream \(syncStatus)")
        cudaDeviceSynchronize()
        #expect((0..<hostArray.count).allSatisfy { outArray[$0] == hostArray[$0] })
    }

    @Test func testCUDAPointerAttribute() async throws {
        let arrayBytes: Int = 64 * MemoryLayout<Float32>.size

        let deviceStatus = CUDevice(index: 0).setDevice()
        #expect(deviceStatus, "Can't set device \(deviceStatus)")

        var aPointer: UnsafeMutableRawPointer?
        let attributesPreAllocation: cudaPointerAttributes = aPointer.getCUDAAttributes()
        #expect(attributesPreAllocation.type == cudaMemoryType.init(0), "Memory is allocated already")

        let allocateStatus = aPointer.cudaMemoryAllocate(arrayBytes)
        #expect(allocateStatus.isSuccessful, "Can't allocate memory for aPointer \(allocateStatus)")

        let attributesPostAllocation: cudaPointerAttributes = aPointer.getCUDAAttributes()
        #expect(attributesPostAllocation.type == cudaMemoryType.init(2), "Memory is not allocated on device")

    }

    @Test func testCUDADealloc() async throws {
        let arrayBytes: Int = 64 * MemoryLayout<Float32>.size

        let deviceStatus = CUDevice(index: 0).setDevice()
        #expect(deviceStatus, "Can't set device \(deviceStatus)")

        var aPointer: UnsafeMutableRawPointer?

        let attributesPreAllocation: cudaPointerAttributes = aPointer.getCUDAAttributes()
        #expect(attributesPreAllocation.type == cudaMemoryType.init(0), "Memory is allocated already")
        #expect(aPointer.isAllocatedOnDevice() == false, "Memory is allocated on device")

        let allocateStatus = aPointer.cudaMemoryAllocate(arrayBytes)
        #expect(allocateStatus.isSuccessful, "Can't allocate memory for aPointer \(allocateStatus)")

        let attributesPostAllocation: cudaPointerAttributes = aPointer.getCUDAAttributes()
        #expect(attributesPostAllocation.type == cudaMemoryType.init(2), "Memory type != cudaMemoryTypeDevice")
        #expect(aPointer.isAllocatedOnDevice() == true, "Memory is not allocated on device")

        let deallocateStatus = aPointer.cudaAndHostDeallocate()
        #expect(deallocateStatus.isSuccessful, "Memory can't be deallocated \(deallocateStatus)")
    }

}
/// This tests should be run on device that isn't using GPU at moment of running tests
/// Tests assume that no memory will be allocated on device memory while running
struct CUDATestMemComplex {
    func testCUDAMemoryInfo() async throws {
        let arrayBytes: Int = Int(pow(2.0, 26.0)) * MemoryLayout<Float32>.size  // ~256mb memory block

        let deviceStatus = CUDevice(index: 0).setDevice()
        #expect(deviceStatus, "Can't set device \(deviceStatus)")

        var emptyPointer: UnsafeMutableRawPointer?
        var aPointer: UnsafeMutableRawPointer?

        defer {
            let _ = emptyPointer.cudaAndHostDeallocate()
            let _ = aPointer.cudaAndHostDeallocate()
        }

        var memory = CUMemory()
        #expect(memory.free == 0 && memory.free == 0)

        let _ = emptyPointer.cudaMemoryAllocate(0)  // empty malloc to update memory
        let _ = cudaDeviceSynchronize()
        var updateStatus = memory.updateCUDAMemory()
        #expect(updateStatus.isSuccessful)
        #expect(memory.free > 0 && memory.free > 0)
        let (free, total) = (memory.free, memory.total)
        let allocateStatus = aPointer.cudaMemoryAllocate(arrayBytes)
        #expect(allocateStatus.isSuccessful, "Can't allocate memory for aPointer \(allocateStatus)")
        updateStatus = memory.updateCUDAMemory()

        #expect(updateStatus.isSuccessful)
        #expect(free > memory.free, "Free memory wasn't bigger before 256mb cuda malloc on device")
        #expect(free - memory.free == arrayBytes, "Difference beetwen memory allocated and free != to 256 mb")
        #expect(total == memory.total)
    }

    func testCUDADeallocWithMemory() async throws {
        let arrayBytes: Int = Int(pow(2.0, 26.0)) * MemoryLayout<Float32>.size

        let deviceStatus = CUDevice(index: 0).setDevice()

        #expect(deviceStatus, "Can't set device \(deviceStatus)")
        var aPointer: UnsafeMutableRawPointer?

        var memory = CUMemory()
        #expect(memory.free == 0 && memory.free == 0)
        let attributesPreAllocation: cudaPointerAttributes = aPointer.getCUDAAttributes()
        #expect(attributesPreAllocation.type == cudaMemoryType.init(0), "Memory is allocated already")
        #expect(aPointer.isAllocatedOnDevice() == false, "Memory is allocated on device")

        let allocateStatus = aPointer.cudaMemoryAllocate(arrayBytes)
        #expect(allocateStatus.isSuccessful, "Can't allocate memory for aPointer \(allocateStatus)")

        var updateStatus = memory.updateCUDAMemory()
        #expect(updateStatus.isSuccessful)
        #expect(memory.free > 0 && memory.free > 0)
        let free = memory.free

        let attributesPostAllocation: cudaPointerAttributes = aPointer.getCUDAAttributes()
        #expect(attributesPostAllocation.type == cudaMemoryType.init(2), "Memory type != cudaMemoryTypeDevice")
        #expect(aPointer.isAllocatedOnDevice() == true, "Memory is not allocated on device")

        let deallocateStatus = aPointer.cudaAndHostDeallocate()
        #expect(deallocateStatus.isSuccessful, "Memory can't be deallocated \(deallocateStatus)")
        updateStatus = memory.updateCUDAMemory()

        #expect(updateStatus.isSuccessful)
        #expect(updateStatus.isSuccessful)
        #expect(memory.free > free, "Free memory wasn't bigger before 256mb cuda malloc on device")
        #expect((memory.free - free) == arrayBytes, "Difference beetwen memory allocated and free != to 256 mb")
    }

    @Test("Sequential memory info tests")
    func complexMemoryTest() async throws {
        try await testCUDAMemoryInfo()
        try await testCUDADeallocWithMemory()
    }
}
