cmake_minimum_required(VERSION 3.12)
project(sealion_inference
    VERSION "0.1"
    DESCRIPTION "Running SEA-LION inference using ggml" 
    HOMEPAGE_URL "https://github.com/jasshanK/sealion_inference")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# deps 
add_subdirectory(ggml)

# add libraries
add_library(common STATIC common.cpp)
target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(common-ggml STATIC common-ggml.cpp)
target_link_libraries(common-ggml PRIVATE ggml)
target_include_directories(common-ggml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# executable 
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC ggml)
target_link_libraries(${PROJECT_NAME} PRIVATE common common-ggml)
target_include_directories(${PROJECT_NAME} PUBLIC .)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

# quantization executable 
option(BUILD_QUANTIZE "Build the quantization executable" ON)

if (BUILD_QUANTIZE)
    add_executable(quantize quantize.cpp)
    target_link_libraries(quantize PUBLIC ggml)
    target_link_libraries(quantize PRIVATE common common-ggml)
    target_include_directories(quantize PUBLIC .)
    target_compile_features(quantize PUBLIC cxx_std_11)
endif()


