# SwiftCU

SwiftCU is a wrapper for CUDA runtime API's (exposed as cxxCU) with extra utilities for device management, memory ops and kernel execution, along with a robust suite of tests.
Repo is tested on newest (v12.5) CUDA runtime API on both Linux and Windows.

| Operating System | Swift Version | CUDA Version | Supported |
|------------------|---------------|--------------|-----------|
| Linux            | 6.0           | 12.*         | ✅        |
| Windows 11       | 6.0           | 12.*         | ✅        |


## Installation

To include SwiftCU in your Swift project, add the following line to your `Package.swift` file:

```swift
.package(url: "https://github.com/machineko/SwiftCU", branch: "main")
```

## Documentation
Docc generated for Swift wrapped API [SwiftCU](https://swiftcu.kobus.me/documentation/swiftcu)

CUDA runtime [CUDART](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)

## Examples

```swift
import SwiftCU

let deviceStatus = CUDADevice(index: 0).setDevice()
if deviceStatus {
    print("Device initialized successfully")
} else {
    print("Failed to initialize device")
}

var pointer: UnsafeMutableRawPointer?
let allocationStatus = pointer.cudaMemoryAllocate(1024)
if allocationStatus.isSuccessful {
    print("Memory allocated successfully")
} else {
    print("Memory allocation failed")
}

let kernel = CUDAKernel(functionPointer: pointerToCUDAKernel)
let launchStatus = kernel.launch(arguments: kernelArgs, blockDim: blockDim, gridDim: gridDim)
if launchStatus.isSuccessful {
    print("Kernel launched successfully")
} else {
    print("Kernel launch failed")
}
```

### For more usage examples check [SwiftCUTests](https://github.com/machineko/SwiftCU/tree/main/Tests/SwiftCUTests) 

## Simple Package overview

```swift
import PackageDescription
import Foundation

let packageDir = URL(fileURLWithPath: #file).deletingLastPathComponent().path
#if os(Windows)
    let cuPath = ProcessInfo.processInfo.environment["CUDA_HOME"] ?? "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5"
    let cuIncludePath = "-I\(cuPath)\\include"
#elseif os(Linux)
    let cuPath = ProcessInfo.processInfo.environment["CUDA_HOME"] ?? "/usr/local/cuda"
    let cuIncludePath = "-I\(cuPath)/include"
#else
    fatalError("OS not supported \(os)")
#endif

let package = Package(
    name: "SwiftCU-example",
    dependencies: 
    [
        .package(url: "https://github.com/machineko/SwiftCU", branch: "main"),
    ],
    targets: [
        .target(
            name: "cudaKernel",
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath(cuIncludePath)
            ]
        ),
        .executableTarget(
            name: "SwiftCU-example",
            dependencies: [
                "cudaKernel",
                .product(name: "SwiftCU", package: "SwiftCU"),
            ],
             swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(
                    ["-Xcc", cuIncludePath]
                )
            ]
        ),
    ]
)
```
## Example build without package/env flags
```fish
swift build -Xswiftc -L/usr/local/cuda/lib64 -Xswiftc -I/usr/local/cuda/include
```


## Testing
Current version of SwiftCU is tested with Swift 6.0 development branch using swift-testing package and CUDA v12.5
