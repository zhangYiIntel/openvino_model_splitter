cmake_minimum_required(VERSION 3.8)
project(ov_test)

set(CMAKE_CXX_STANDARD 11)
find_package(InferenceEngine REQUIRED)
add_executable(ov_model_splitter ov_model_splitter.cpp)

target_include_directories(ov_model_splitter PUBLIC ${InferenceEngine_INCLUDE_DIRS})
target_link_libraries(ov_model_splitter ${InferenceEngine_LIBRARIES})


