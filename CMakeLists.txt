cmake_minimum_required(VERSION 3.2)

#编译宏的设定，可通过如下方式进行开关的控制，示例：
#$cmake ../ -=USE_TENSORRT=ON -DUSE_DEBUG=ON
#编译宏开关初期化
option (USE_DEBUG           "Set switch to build at Debug mode"             OFF)
option (USE_CAFFE           "Set switch to build at USE_CAFFE mode"         OFF)
option (USE_TENSORRT        "Set switch to build at USE_TENSORRT mode"      ON)
option (USE_NPP             "Set switch to build at USE_NPP mode"           OFF)


###############
#工程概述
###############
PROJECT (goggleMaskRT)
MESSAGE (STATUS "Project: goggleMaskRT ")  # retinaFaceReImp
MESSAGE (STATUS "Project Directory: ${PROJECT_SOURCE_DIR}")

###############
#环境变量
###############
#编译器环境变量设定
add_definitions (-std=c++11 -O2 -fomit-frame-pointer -g -Wall)
MESSAGE (STATUS "Build Option: -std=c++11 -O2 -fomit-frame-pointer -g -Wall")


find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(PkgConfig)
# find protobuf3.x
pkg_search_module(PROTOBUF REQUIRED protobuf)

#if(USE_TENSORRT)
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
#        -O3 -gencode arch=compute_50,code=sm_50;
#        -gencode arch=compute_52,code=sm_52;
#        -gencode arch=compute_53,code=sm_53;
#        -gencode arch=compute_60,code=sm_60;
#        -gencode arch=compute_61,code=sm_61;
#        -gencode arch=compute_62,code=sm_62;
#        -gencode arch=compute_70,code=sm_70;
#    )
#endif()


#模式： RELEASE/DEBUG 
#默认： RELEASE
if(USE_DEBUG)
    add_definitions(-D_DEBUG)
    MESSAGE (STATUS "Build Option: -D_DEBUG")
endif()

#模式： TENSORRT/CAFFE
#默认： TENSORRT

if(USE_TENSORRT)
    add_definitions(-DUSE_TENSORRT)
    MESSAGE (STATUS "Build Option: -DUSE_TENSORRT")
elseif(USE_CAFFE)
    add_definitions(-DUSE_CAFFE)
    MESSAGE (STATUS "Build Option: -DUSE_CAFFE")
endif()

if(USE_NPP)
    add_definitions(-DUSE_NPP)
    MESSAGE(STATUS "Build Option: -DUSE_NPP")
endif()

###############
#添加头文件
###############
# set caffe path
set(CAFFE_ROOT $ENV{HOME}/caffe-master)
# message(STATUS "CAFFE_ROOT: ${CAFFE_ROOT}")   # /home/hbk/Tool/caffe_gpu
set(CAFFE_INCLUDE_DIRS ${CAFFE_ROOT}/build/src
                       ${CAFFE_ROOT}/include)
set(CAFFE_LIB_DIRS ${CAFFE_ROOT}/build/lib)
set(TENSORRT_ROOT $ENV{HOME}/software/TensorRT-6.0.1.5)
include_directories (
    "./goggleMask"
    "./goggleMask/tensorrt"
    ${PROTOBUF_INCLUDE_DIRS}
    ${OpenCV_INLCUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CAFFE_INCLUDE_DIRS}
    ${TENSORRT_ROOT}/include
)

###############
#添加源文件
###############
AUX_SOURCE_DIRECTORY(./goggleMask DIR_SRCS)

message(STATUS "source file : ${DIR_SRCS}")


###############
#添加引用类库
###############
if(USE_TENSORRT)
    link_directories(
        ${PROTOBUF_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${CAFFE_LIB_DIRS}
        ${TENSORRT_ROOT}/lib)
elseif(USE_CAFFE)
    link_directories(
        ${PROTOBUF_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${CAFFE_LIB_DIRS})
endif()

###############
#生成demo
###############

if(USE_TENSORRT)
    # if(USE_NPP)
    #     file( GLOB  core_cuda_files  "./goggleMask/*.cu")
    # endif()
    AUX_SOURCE_DIRECTORY(./goggleMask/tensorrt DIR_SRCS_CUDA)
    cuda_add_executable(goggleMask ${DIR_SRCS} ${DIR_SRCS_CUDA} ${core_cuda_files})

    list(REMOVE_ITEM ./goggleMask/main.cpp ${DIR_SRCS} )
    add_library(gogglemask SHARED ${DIR_SRCS} ${DIR_SRCS_CUDA} ${core_cuda_files})

    target_link_libraries(goggleMask 
        ${OpenCV_LIBRARIES} 
        caffe glog boost_system 
        cudnn cublas cudart
        nvcaffe_parser nvinfer
        nppig nppicc nppc nppidei nppist)
else()
    add_executable(goggleMask ${DIR_SRCS})
    target_link_libraries(goggleMask 
        ${OpenCV_LIBRARIES} 
        caffe glog boost_system 
        cudnn cublas cudart)
endif()

