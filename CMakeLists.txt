cmake_minimum_required(VERSION 3.10)

project(yolov8)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)

# Modify to your path
set(OpenCV_DIR "E:/ai/opencv 3.4.16/opencv/build")
set(TRT_DIR "E:/ai/TensorRT-8.6.1")
set(EIGEN3 "C:/Program Files (x86)/Eigen3")

add_definitions(-DAPI_EXPORTS)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${OpenCV_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)
include_directories(${EIGEN3}/include/eigen3)

# TensorRT
#find_package(TRT_DIR)
include_directories(${TRT_DIR}/include)
link_directories(${TRT_DIR}/lib)


# setup CUDA
find_package(CUDA REQUIRED)
message(STATUS "libraries: ${CUDA_LIBRARIES}")
message(STATUS "include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-g;-G;-gencode;arch=compute_86;code=sm_86)

enable_language(CUDA)

# OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart ${OpenCV_LIBS})

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
add_executable(yolov8 ${PROJECT_SOURCE_DIR}/src/main.cpp ${SRCS})

target_link_libraries(yolov8 nvinfer)
target_link_libraries(yolov8 cudart)
target_link_libraries(yolov8 myplugins)
target_link_libraries(yolov8 ${OpenCV_LIBS})
