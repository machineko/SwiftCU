// swift-tools-version: 6.0

import PackageDescription
import Foundation

let packageDir = URL(fileURLWithPath: #file).deletingLastPathComponent().path
#if os(Windows)
    let cuPath = ProcessInfo.processInfo.environment["CUDA_HOME"] ?? "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5"
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
            .package(url: "https://github.com/apple/swift-testing.git", from: "0.10.0"),
            .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0"),

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
    cxxLanguageStandard: .cxx17
)

/// usage 
/// CUDA_HOME=/usr/local/cuda swift test 
/// To turn on safetyCheck add -Xswiftc -DsafetyCheck 
/// Test on rtx 3090 -Xswiftc -Drtx3090Test -Xcc -Drtx3090Test -Xswiftc -L$(pwd)/Sources/cxxCU/lib -Xswiftc -lcuADD
/// basic test flags -Xswiftc -L/usr/local/cuda/lib64 -Xswiftc -I/usr/local/cuda/include
