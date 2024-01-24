cmake_minimum_required(VERSION 3.8)
project(ov_test)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenVINO REQUIRED)
add_executable(ov_model_splitter ov_model_splitter.cpp)

target_link_libraries(ov_model_splitter openvino::runtime)


