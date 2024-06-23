import Testing

@testable import SwiftCU

struct DeviceTest {
    @Test func testDeviceInit() throws {
        let cuStatus = CUDevice(index: 0).setDevice()
        print(cuStatus)
        #expect(cuStatus)
    }

    @Test func testDeviceInitFail() throws {
        let cuStatus = CUDevice(index: -1).setDevice()
        #expect(!cuStatus)
    }

    #if rtx3090Test
        @Test func testRTX3090Properties() throws {
            let device = CUDevice(index: 0)
            let prop = device.getDeviceProperties()
            #expect(prop.deviceName == "NVIDIA GeForce RTX 3090")
            #expect(prop.major == 8)
            #expect(prop.minor == 6)
        }
    #endif
}
