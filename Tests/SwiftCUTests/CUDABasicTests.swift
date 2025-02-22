import Testing

@testable import SwiftCU

struct DeviceTest {

    @Test func testDeviceInit() async throws {
        let cuStatus = CUDADevice(index: 0).setDevice()
        #expect(cuStatus)
    }

    #if (!safetyCheck)
        @Test func testDeviceInitFail() async throws {
            let cuStatus = CUDADevice(index: -1).setDevice()
            #expect(!cuStatus)
        }
    #endif

    @Test func testCUDAStreamCreation() async throws {
        let cuStatus = CUDADevice(index: 0).setDevice()
        #expect(cuStatus)
        let stream = cudaStream()
        #expect(stream.stream != nil)
    }

    #if rtx3090Test
        @Test func testRTX3090Properties() async throws {
            let device = CUDADevice(index: 0)
            let prop = device.getDeviceProperties()
            #expect(prop.deviceName == "NVIDIA GeForce RTX 3090")
            #expect(prop.major == 8)
            #expect(prop.minor == 6)
        }
    #endif
}
