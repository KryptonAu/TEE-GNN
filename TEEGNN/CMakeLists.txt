cmake_minimum_required(VERSION 3.10)

project(teegnn_main LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenSSL REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

file(GLOB C_FILES "src/*.c")
file(GLOB CPP_FILES "host/src/*.cpp")
file(GLOB TEEC_FILES "../third_party/optee_client/libteec/src/*.c")

add_library(ta
    ${CMAKE_CURRENT_SOURCE_DIR}/ta/teegnn_ta.c
)

target_include_directories(ta
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        /home/ka/TEE-GNN/third_party/out/data/link/export-ta_arm64/include
)


add_library(teegnn
    ${C_FILES}
    ${CPP_FILES}
    ${TEEC_FILES}
)

target_include_directories(teegnn
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/host/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/optee_client/libteec/include
)

target_link_libraries(teegnn
    PUBLIC
        OpenSSL::Crypto
        Eigen3::Eigen
)

add_executable(teegnn_main
    host/teegnn_main.cpp
)

target_link_libraries(teegnn_main
    PRIVATE
        teegnn
)

