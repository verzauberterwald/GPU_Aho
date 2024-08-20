cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(aho_app LANGUAGES CXX CUDA)

find_package(OpenCL REQUIRED)

add_executable(app src/cpu.cpp src/base.cpp src/read_helper.cpp src/vector_builder.cpp)
add_executable(app_opencl src/opencl.cpp src/base.cpp src/read_helper.cpp src/vector_builder.cpp)
add_executable(app_cuda src/cuda.cu src/base.cpp src/read_helper.cpp src/vector_builder.cpp)

# Link the OpenCL libraries
target_link_libraries(app_opencl PUBLIC OpenCL::OpenCL)