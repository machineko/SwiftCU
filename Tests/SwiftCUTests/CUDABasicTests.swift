import Testing
@testable import SwiftCU

struct DeviceTest {
    @Test func testDeviceInit() async throws {
        let cuStatus = CUDevice(index: 0).setDevice()
        #expect(cuStatus)
    }

    @Test func testDeviceInitFail() async throws {
        let cuStatus = CUDevice(index: -1).setDevice()
        #expect(!cuStatus)
    }

    @Test func testCUDAStreamCreation() async throws {
        let cuStatus = CUDevice(index: 0).setDevice()
        #expect(cuStatus)
        let stream = cudaStream()
        #expect(stream.stream != nil)
    }



    #if rtx3090Test
        @Test func testRTX3090Properties() async throws {
            let device = CUDevice(index: 0)
            let prop = device.getDeviceProperties()
            #expect(prop.deviceName == "NVIDIA GeForce RTX 3090")
            #expect(prop.major == 8)
            #expect(prop.minor == 6)
        }
    #endif
}

