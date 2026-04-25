cmake_minimum_required(VERSION 3.14)
project(teegnn_nontee_sim LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

add_executable(teegnn_sim
    teegnn_sim_main.cpp
    ../common/src/csprng_adapter.cpp
    ../common/src/dataset_loader.cpp
    ../common/src/gcn_ops.cpp
    ../common/src/graph.cpp
    ../common/src/masks.cpp
    ../CSPRNG/csprng.c
)

target_include_directories(teegnn_sim PRIVATE
    ../common/include
    ../CSPRNG
)
target_link_libraries(teegnn_sim PRIVATE Eigen3::Eigen)
target_compile_options(teegnn_sim PRIVATE -Wall -Wextra -pedantic)

