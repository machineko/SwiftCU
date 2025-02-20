cmake_minimum_required(VERSION 3.25)

set(CMAKE_CUDA_ARCHITECTURES
    60-real  # Pascal
    70-real  # Volta
    75-real  # Turing
    80-real  # Ampere (A100)
    86-real  # Ampere (RTX 30xx)
    89-real  # Ada Lovelace
    90-real  # Hopper
)

project(MyCudaProject LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_EXTENSIONS OFF) 
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler '-fPIC'")

file(GLOB CUDA_FILES "Sources/cxxCU/*.cu")

add_library(cudaObjLib OBJECT ${CUDA_FILES})
set_target_properties(cudaObjLib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_library(cuADD STATIC
    $<TARGET_OBJECTS:cudaObjLib>
)

set_target_properties(cuADD PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/Sources/cxxCU/lib"
)

target_link_libraries(cuADD PUBLIC
    CUDA::cudart_static
    CUDA::cuda_driver
)

target_include_directories(cuADD PUBLIC 
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_options(cuADD PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-code=arch=compute_60,code=[compute_60,sm_60]
        --generate-code=arch=compute_70,code=[compute_70,sm_70]
        --generate-code=arch=compute_75,code=[compute_75,sm_75]
        --generate-code=arch=compute_80,code=[compute_80,sm_80]
        --generate-code=arch=compute_86,code=[compute_86,sm_86]
        --generate-code=arch=compute_89,code=[compute_89,sm_89]
        --generate-code=arch=compute_90,code=[compute_90,sm_90]
    >
)
