import re
import os
from pathlib import Path

def read_cuda_error_enum(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    enum_match = re.search(r'enum __device_builtin__ cudaError\s*{([^}]*)}', content)

    if enum_match is None:
        print("Enum 'cudaError' not found in file.")
        return None
    enum_content = enum_match.group(1)
    enum_content = re.sub(r'\n\s*,', '\n', enum_content)
    return str(enum_content)

def convert_cpp_enum_to_swift(cpp_enum):
    lines = cpp_enum.split('\n')
    swift_enum = "public enum cudaError: Int {\n"
    docstring = ""
    for line in lines:
        if "enum" in line or "{" in line or "}" in line:
            continue
        if "=" in line:
            parts = line.split('=')
            enum_name = parts[0].strip()
            enum_value = parts[1].split(',')[0].strip()
            swift_enum += f"{docstring}    case {enum_name} = {enum_value}\n"
            docstring = ""
        elif "/**" in line:
            docstring = line.replace("/**", "///").strip() + "\n"
        elif "*/" in line:
            continue
        else:
            docstring += line.replace("* ", "/// ").strip() + "\n"
    swift_enum += "}"
    return swift_enum

def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)


def get_cuda_driver_path():
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home is None:
        raise Exception('CUDA_HOME environment variable is not set')
    return os.path.join(cuda_home, 'include', 'driver_types.h')

if __name__ == "__main__":
    cuda_error = read_cuda_error_enum(get_cuda_driver_path())
    if cuda_error:
        swift_enum = convert_cpp_enum_to_swift(cuda_error)
        write_to_file(Path("Sources/SwiftCU/CUDATypes/CUDAError.swift"), swift_enum)
        file_path = os.path.normpath("Sources/SwiftCU/CUDATypes/CUDAError.swift")
    else:
        raise FileNotFoundError(f"Can't find cudaError file {get_cuda_driver_path()}")
    