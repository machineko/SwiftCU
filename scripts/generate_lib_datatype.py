import re
import os
from pathlib import Path

def read_cublas_enum(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def convert_cpp_enum_to_swift(cpp_enum):
    enums = re.findall(r'typedef\s+enum\s+\w*\s*\{([^}]*)\}\s*(\w+);', cpp_enum, re.DOTALL)
    lines = enums[0][0].split('\n')
    swift_enum = "public enum SwifCUDADataType: Int {\n"
    docstring = ""
    for line in lines:
        line = line.strip()
        if not line or "{" in line or "}" in line:
            continue
        if "=" in line:
            parts = line.split('=')
            enum_name = parts[0].strip()
            enum_value = parts[1].split(',')[0].strip()

            swift_enum += f"    case {enum_name} = {enum_value} {docstring}\n"
            if docstring:
                # swift_enum += f"{docstring}"
                docstring = ""
        if "/*" in line:
            docstring = "    //" + line.split("/*")[1].strip().replace("*/", "")
        elif "*" in line:
            docstring = "    //" + line.split("*")[1].strip()
    swift_enum += "}"
    return swift_enum

def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def get_cublas_header_path():
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home is None:
        raise Exception('CUDA_HOME environment variable is not set')
    return os.path.join(cuda_home, 'include', 'library_types.h')

if __name__ == "__main__":
    cublas_header_path = get_cublas_header_path()
    cublas_enum_content = read_cublas_enum(cublas_header_path)
    swift_enum_content = convert_cpp_enum_to_swift(cublas_enum_content)
    output_file_path = Path("Sources/SwiftCU/CUDATypes/CUDADataTypeEnum.swift")
    write_to_file(output_file_path, swift_enum_content)
    os.system(f"swift-format {output_file_path} -i {output_file_path}")
