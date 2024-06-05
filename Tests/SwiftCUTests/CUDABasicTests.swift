import XCTest
@testable import SwiftCU

final class SwiftCUTests: XCTestCase {
    func testDeviceInit() throws {
        let cuStatus = CUDevice(index: 0).setDevice()
        print(cuStatus)
        XCTAssert(cuStatus)
    }

    func testDeviceInitFail() throws {
        let cuStatus = CUDevice(index: -1).setDevice()
        XCTAssert(!cuStatus)
    }
}
