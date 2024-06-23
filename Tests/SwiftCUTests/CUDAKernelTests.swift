import Testing

@testable import SwiftCU
@testable import cxxCu

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
            // print("before call: tempPointer = \(String(describing: aPointer))")
            // var allocateStatus = aPointer.cudaMemoryAllocate(arraySize)
            // print("After call: tempPointer = \(String(describing: aPointer))")

            // #expect(allocateStatus.isSuccessful, "Can't allocate memory for aPointer \(allocateStatus)")
            // allocateStatus = bPointer.cudaMemoryAllocate(arraySize)
            // #expect(allocateStatus.isSuccessful, "Can't allocate memory for bPointer \(allocateStatus)")
            // allocateStatus = cPointer.cudaMemoryAllocate(arraySize)
            // #expect(allocateStatus.isSuccessful, "Can't allocate memory for cPointer \(allocateStatus)")
            print("before call: tempPointer = \(String(describing: aPointer))")

            cudaMalloc(&aPointer, arrayBytes)

            print("after call: tempPointer = \(String(describing: aPointer))")

            // cudaMalloc(&bPointer, arrayBytes)
            // cudaMalloc(&cPointer, arrayBytes)
            var (a, b) = (
                [Float32](repeating: 0.0, count: arraySize), [Float32](repeating: 0.0, count: arraySize)
            )
            for i in 0..<32 {
                a[i] = Float32(i)
                b[i] = Float32(i * 2)
            }
            var copyStatus = cudaMemcpy(aPointer, &a, arrayBytes, cudaMemcpyHostToDevice).asSwift
            #expect(copyStatus.isSuccessful, "Can't copy memory for aPointer \(copyStatus)")
            copyStatus = cudaMemcpy(bPointer, &b, arrayBytes, cudaMemcpyHostToDevice).asSwift
            #expect(copyStatus.isSuccessful, "Can't copy memory for bPointer \(copyStatus)")
            

            var kernelArgs = CUDAKernelArguments()
            // kernelArgs.addRawPointer(&aPointer)
            kernelArgs.addRawPointers([aPointer, bPointer, cPointer])


            var someValue = Int32(arraySize)
            kernelArgs.addPointer(&someValue)
            let threadsPerBlock = 256
            let blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock
            let blockDim = dim3(make_uint3(UInt32(threadsPerBlock), 1, 1))
            let gridDim = dim3(make_uint3(UInt32(blocksPerGrid), 1, 1))

 
        }
    #endif
}