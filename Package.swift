// swift-tools-version: 5.10

import PackageDescription
import Foundation

let packageDir = URL(fileURLWithPath: #file).deletingLastPathComponent().path

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
    targets: [
        .target(
            name: "cxxCu",
            publicHeadersPath: "include",
            cxxSettings: [
                .unsafeFlags([
                    cuIncludePath,
                ])
            ],
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
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
        .testTarget(
            name: "SwiftCUTests",
            dependencies: ["SwiftCU", "cxxCu"]),
    ]
)
