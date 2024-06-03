import XCTest
@testable import SwiftCU

final class SwiftCUTests: XCTestCase {
    func testDeviceInit() throws {
        let cuStatus = CUDevice(index: 0).setDevice()
        XCTAssert(cuStatus)
    }
}
