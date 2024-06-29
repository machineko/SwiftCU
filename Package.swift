// swift-tools-version: 6.0
// To turn on safetyCheck add -Xswiftc -DsafetyCheck

import PackageDescription
import Foundation

let packageDir = URL(fileURLWithPath: #file).deletingLastPathComponent().path
#if os(Windows)
    let cuPath = ProcessInfo.processInfo.environment["CUDA_HOME"] ?? "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4"
    let cuLibPath = "-L\(cuPath)\\lib\\x64"
    let cuIncludePath = "-I\(cuPath)\\include"
    let cxxLibPath = "-L\(packageDir)\\Sources\\cxxCU\\lib\\Release"
#elseif os(Linux)
    let cuPath = ProcessInfo.processInfo.environment["CUDA_HOME"] ?? "/usr/local/cuda"
    let cuLibPath = "-L\(cuPath)/lib64"
    let cuIncludePath = "-I\(cuPath)/include"
    let cxxLibPath = "-L\(packageDir)/Sources/cxxCU/lib"
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
            name: "cxxCU",
            targets: ["cxxCU"]),
    ],
    dependencies: [
            .package(url: "https://github.com/apple/swift-testing.git", from: "0.2.0")
        ],
    targets: [
        .target(
            name: "cxxCU",
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath(cuIncludePath)
            ],
            linkerSettings: [
                .unsafeFlags([
                    cuLibPath,
                    cxxLibPath,
                ]),
                .linkedLibrary("cudart"),
            ]
        ),
        .target(
            name: "SwiftCU",
            dependencies: ["cxxCU"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(["-Xcc", cuIncludePath]),
            ]
        ),
        .testTarget(
            name: "SwiftCUTests",
           
            dependencies: [
                "SwiftCU", "cxxCU",
                .product(name: "Testing", package: "swift-testing"), 

            ],
             swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        )
    ],
    cxxLanguageStandard: .cxx17 // Only working with nil otherwise it tires to build C over Cxx for test target?
)

///flags 
/// -Xswiftc -L/usr/local/cuda/lib64 -Xswiftc -I/usr/local/cuda/include -Xswiftc -L$(pwd)/Sources/cxxCU/lib -Xswiftc -Drtx3090Test -Xswiftc -lcuADD
