cmake_minimum_required(VERSION 3.14)
project(teegnn_plaintext LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

add_executable(gcn_infer
    plaintext_main.cpp
    ../common/src/dataset_loader.cpp
    ../common/src/graph.cpp
    ../common/src/gcn_ops.cpp
)

target_include_directories(gcn_infer PRIVATE
    ../common/include
)
target_link_libraries(gcn_infer PRIVATE Eigen3::Eigen)
target_compile_options(gcn_infer PRIVATE -Wall -Wextra -pedantic)

