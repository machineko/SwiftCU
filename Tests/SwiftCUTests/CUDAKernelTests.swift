import Foundation
import Testing

@testable import SwiftCU
@testable import cxxCU

struct KernelTest {

    #if rtx3090Test
        @Test func testAddKernel() async throws {
            let arraySize: Int = 32
            let arrayBytes: Int = arraySize * MemoryLayout<Float32>.size
            let deviceStatus = CUDevice(index: 0).setDevice()
            #expect(deviceStatus, "Can't set device \(deviceStatus)")

            var aPointer: UnsafeMutableRawPointer?
            var bPointer: UnsafeMutableRawPointer?
            var cPointer: UnsafeMutableRawPointer?

            var allocateStatus = aPointer.cudaMemoryAllocate(arrayBytes)
            #expect(allocateStatus.isSuccessful, "Can't allocate memory for aPointer \(allocateStatus)")
            allocateStatus = bPointer.cudaMemoryAllocate(arrayBytes)
            #expect(allocateStatus.isSuccessful, "Can't allocate memory for bPointer \(allocateStatus)")
            allocateStatus = cPointer.cudaMemoryAllocate(arrayBytes)
            #expect(allocateStatus.isSuccessful, "Can't allocate memory for cPointer \(allocateStatus)")

            var (a, b) = (
                [Float32](repeating: 0.0, count: arraySize), [Float32](repeating: 0.0, count: arraySize)
            )
            for i in 0..<32 {
                a[i] = Float32(i)
                b[i] = Float32(i * 2)
            }
            var copyStatus = aPointer.cudaMemoryCopy(fromRawPointer: &a, numberOfBytes: arrayBytes, copyKind: .cudaMemcpyHostToDevice)
            #expect(copyStatus.isSuccessful, "Can't copy memory for aPointer \(copyStatus)")
            copyStatus = bPointer.cudaMemoryCopy(fromRawPointer: &b, numberOfBytes: arrayBytes, copyKind: .cudaMemcpyHostToDevice)
            #expect(copyStatus.isSuccessful, "Can't copy memory for bPointer \(copyStatus)")

            var kernelArgs: CUDAKernelArguments = CUDAKernelArguments()
            kernelArgs.addRawPointers([aPointer, bPointer, cPointer])

            var someValue = Int32(arraySize)
            kernelArgs.addPointer(&someValue)
            let threadsPerBlock = 256
            let blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock
            let blockDim = dim3(make_uint3(UInt32(threadsPerBlock), 1, 1))
            let gridDim = dim3(make_uint3(UInt32(blocksPerGrid), 1, 1))
            var syncStatus = cudaDeviceSynchronize()
            #expect(syncStatus.asSwift.isSuccessful, "Can't sync device \(syncStatus)")
            let kernel = CUDAKernel(functionPointer: getKernelPointer(ADD_F32))
            let launchStatus = kernel.launch(arguments: kernelArgs, blockDim: blockDim, gridDim: gridDim)

            #expect(launchStatus.isSuccessful, "Can't launch kernel \(launchStatus)")

            syncStatus = cudaDeviceSynchronize()
            #expect(syncStatus.asSwift.isSuccessful, "Can't sync device \(syncStatus)")

            var outputData = [Float32](repeating: 0, count: arraySize)
            outputData.withUnsafeMutableBytes { rawBufferPointer in
                var pointerAddress = rawBufferPointer.baseAddress
                let outStatus = pointerAddress.cudaMemoryCopy(
                    fromMutableRawPointer: cPointer, numberOfBytes: arrayBytes, copyKind: .cudaMemcpyDeviceToHost)
                #expect(outStatus.isSuccessful, "Can't copy memory from device \(syncStatus)")
            }
            defer {
                var deallocateStatus = aPointer.cudaAndHostDeallocate()
                #expect(aPointer == nil)
                #expect(deallocateStatus.isSuccessful, "Can't dealocate aPointer \(deallocateStatus)")
                deallocateStatus = bPointer.cudaAndHostDeallocate()
                #expect(deallocateStatus.isSuccessful, "Can't dealocate bPointer \(deallocateStatus)")
                deallocateStatus = cPointer.cudaAndHostDeallocate()
                #expect(deallocateStatus.isSuccessful, "Can't dealocate cPointer \(deallocateStatus)")
            }

            for i in 0..<32 {
                let expectedValue = a[i] + b[i]
                #expect(
                    outputData[i] == a[i] + b[i], "results of addition not equal to expected value \(outputData[i]) != \(expectedValue)")
            }
        }
    #endif
}

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
