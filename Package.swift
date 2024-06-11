// swift-tools-version: 6.0
// To turn on safetyCheck add -Xswiftc -DsafetyCheck flag

import PackageDescription
import Foundation
// let packageDir = URL(fileURLWithPath: #file).deletingLastPathComponent().path

#if os(Windows)
    let cuPath = ProcessInfo.processInfo.environment["CUDA_HOME"] ?? "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4"
    let cuLibPath = "-L\(cuPath)\\lib\\x64"
    let cuIncludePath = "-I\(cuPath)\\include"
#elseif os(Linux)
    let cuPath = ProcessInfo.processInfo.environment["CUDA_HOME"] ?? "/usr/local/cuda"
    let cuLibPath = "-L\(cuPath)/lib64"
    let cuIncludePath = "-I\(cuPath)/include"
#else
    fatalError("OS not supported \(os)")
#endif

let package = Package(
    name: "SwiftCU",
    products: [
        .library(
            name: "SwiftCU",
            targets: ["SwiftCU"]),
        .library(
            name: "cxxCu",
            targets: ["cxxCu"]),
    ],
    dependencies: [.package(url: "https://github.com/apple/swift-testing.git", branch: "main")],
    targets: [
        .target(
            name: "cxxCu",
            publicHeadersPath: "include",
            linkerSettings: [
                .unsafeFlags([
                    cuLibPath,
                    "-lcudart",
                ])
            ]
        ),
        .target(
            name: "SwiftCU",
            dependencies: ["cxxCu"],
             cxxSettings: [
                .unsafeFlags([
                    cuIncludePath,
                ])
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(
                    [
                        "-Xcc", cuIncludePath,
                        "-lcudart",
                    ]
                ),
            ]
        ),
        .testTarget(
            name: "SwiftCUTests",
            dependencies: [.product(name: "Testing", package: "swift-testing"), "SwiftCU", "cxxCu"]
        ),
            
    ],
    cxxLanguageStandard: nil // Only working with nil otherwise it tires to build C over Cxx for test target?
)
