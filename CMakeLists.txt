cmake_minimum_required(VERSION 3.20)
project(ov_test)

set(CMAKE_CXX_STANDARD 11)
find_package(InferenceEngine REQUIRED)
add_executable(ov_test main.cpp)
add_executable(matmul_test matmul_test.cpp)
add_executable(ov_model_splitter ov_model_splitter.cpp)
target_include_directories(ov_test PUBLIC ${InferenceEngine_INCLUDE_DIRS})
target_link_libraries(ov_test ${InferenceEngine_LIBRARIES})

target_include_directories(matmul_test PUBLIC ${InferenceEngine_INCLUDE_DIRS})
target_link_libraries(matmul_test ${InferenceEngine_LIBRARIES})


target_include_directories(ov_model_splitter PUBLIC ${InferenceEngine_INCLUDE_DIRS})
target_link_libraries(ov_model_splitter ${InferenceEngine_LIBRARIES})