import Testing
@testable import SwiftCU
@testable import cxxCU

struct KernelTest {
    #if rtx3090Test
        @Test func testAddKernel() throws {
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
            

            var kernelArgs = CUDAKernelArguments()
            kernelArgs.addRawPointers([aPointer, bPointer, cPointer])


            var someValue = Int32(arraySize)
            kernelArgs.addPointer(&someValue)
            let threadsPerBlock = 256
            let blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock
            let blockDim = dim3(make_uint3(UInt32(threadsPerBlock), 1, 1))
            let gridDim = dim3(make_uint3(UInt32(blocksPerGrid), 1, 1))
            let launchStatus = cudaLaunchKernel(
                getKernelPointer(ADD_F32), gridDim, blockDim, kernelArgs.getArgsPointer(), 0, nil
            )
            #expect(launchStatus.asSwift.isSuccessful, "Can't launch kernel \(launchStatus)")

            let syncStatus = cudaDeviceSynchronize()
            #expect(syncStatus.asSwift.isSuccessful, "Can't sync device \(syncStatus)")

            var outputData = [Float32](repeating: 0, count: arraySize)
            var outputPointer: UnsafeMutableRawPointer? = outputData.withUnsafeMutableBytes { rawBufferPointer in
                return rawBufferPointer.baseAddress
            }
            let outStatus = outputPointer.cudaMemoryCopy(fromMutableRawPointer: cPointer, numberOfBytes: arrayBytes, copyKind: .cudaMemcpyDeviceToHost)
            for i in 0..<32 {
                let expectedValue = a[i] + b[i]
                #expect(outputData[i] == a[i] + b[i], "results of addition not equal to expected value \(outputData[i]) != \(expectedValue)")
            }    
        }
    #endif
}