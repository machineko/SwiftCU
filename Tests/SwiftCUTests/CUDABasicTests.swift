@testable import SwiftCU
import Testing

struct DeviceTest {
    @Test func testDeviceInit() throws {
        let cuStatus = CUDevice(index: 0).setDevice()
        print(cuStatus)
        #expect(cuStatus)
    }

    @Test  func testDeviceInitFail() throws {
        let cuStatus = CUDevice(index: -1).setDevice()
        #expect(!cuStatus)
    }
}