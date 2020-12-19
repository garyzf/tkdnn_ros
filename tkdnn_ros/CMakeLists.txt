cmake_minimum_required(VERSION 2.8.12)
project(tkdnn_ros)

# Set c++11 cmake flags
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wno-deprecated-declarations -Wno-unused-variable")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Define path of tkDNN folder here.
find_path(TKDNN_PATH
  NAMES "README.md"
  HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../tkDNN/")
message(STATUS "TkDNN path dir = ${TKDNN_PATH}")
add_definitions(-DTKDNN_FILE_PATH="${TKDNN_PATH}")
add_definitions(-DTKDNN_PATH="${TKDNN_PATH}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${TKDNN_PATH}/cmake)
message(STATUS "cmake path dir = ${CMAKE_MODULE_PATH}")
include_directories(${TKDNN_PATH}/include/tkDNN)

# Find CUDA
find_package(CUDA QUIET)
SET(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --maxrregcount=32)

# find CUDNN
find_package(CUDNN REQUIRED)
include_directories(${CUDNN_INCLUDE_DIR})

# Find Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Find Boost
find_package(Boost REQUIRED COMPONENTS thread)

# Find OpenCV
find_package(OpenCV 3.3.1 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV")
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(catkin REQUIRED
  COMPONENTS
    cv_bridge
    roscpp
    rospy
    sensor_msgs
    std_msgs
    tkdnn_ros_msgs
    actionlib
    image_transport
    tf
)

catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}_lib
  CATKIN_DEPENDS
    cv_bridge
    roscpp
    actionlib
    rospy
    std_msgs
    tkdnn_ros_msgs
    image_transport
  DEPENDS
    Boost
)

include_directories(
  ${TKDNN_PATH}/src
  ${TKDNN_PATH}/src/kernels
  ${TKDNN_PATH}/include/tkDNN
  ${TKDNN_PATH}/include
  ${TKDNN_PATH}/tests/darknet
  include
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_definitions(-DOPENCV)
add_definitions(-O4 -g)

# tkDNN
set(TKDNN_CUDA_FILES
    ${TKDNN_PATH}/src/kernels/activation_elu.cu           ${TKDNN_PATH}/src/kernels/convert.cu
    ${TKDNN_PATH}/src/kernels/activation_leaky.cu         ${TKDNN_PATH}/src/kernels/deformable_conv.cu
    ${TKDNN_PATH}/src/kernels/activation_logistic.cu      ${TKDNN_PATH}/src/kernels/fill.cu
    ${TKDNN_PATH}/src/kernels/activation_mish.cu          ${TKDNN_PATH}/src/kernels/normalize.cu
    ${TKDNN_PATH}/src/kernels/activation_relu_ceiling.cu  ${TKDNN_PATH}/src/kernels/pooling.cu
    ${TKDNN_PATH}/src/kernels/activation_sigmoid.cu       ${TKDNN_PATH}/src/kernels/postprocessing.cu
    ${TKDNN_PATH}/src/kernels/reorg.cu                    ${TKDNN_PATH}/src/kernels/resize.cu
    ${TKDNN_PATH}/src/kernels/scaladd.cu                  ${TKDNN_PATH}/src/kernels/shortcut.cu
    ${TKDNN_PATH}/src/kernels/softmax.cu                  ${TKDNN_PATH}/src/kernels/upsample.cu
   # ${TKDNN_PATH}/src/sorting.cu
)
cuda_include_directories(${TKDNN_PATH}/include ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS})
cuda_add_library(kernels SHARED ${TKDNN_CUDA_FILES})

set(TKDNN_CORE_FILES
    ${TKDNN_PATH}/src/Activation.cpp                     ${TKDNN_PATH}/src/BoundingBox.cpp
    ${TKDNN_PATH}/src/CenternetDetection.cpp             ${TKDNN_PATH}/src/Conv2d.cpp
    ${TKDNN_PATH}/src/DarknetParser.cpp                  ${TKDNN_PATH}/src/DarknetParser.cpp
    ${TKDNN_PATH}/src/DeformConv2d.cpp                   ${TKDNN_PATH}/src/Conv2d.cpp
    ${TKDNN_PATH}/src/Dense.cpp                          ${TKDNN_PATH}/src/Flatten.cpp
    ${TKDNN_PATH}/src/evaluation.cpp                     ${TKDNN_PATH}/src/Int8BatchStream.cpp
    ${TKDNN_PATH}/src/Int8Calibrator.cpp                 ${TKDNN_PATH}/src/Layer.cpp
    ${TKDNN_PATH}/src/LayerWgs.cpp                       ${TKDNN_PATH}/src/LSTM.cpp
    ${TKDNN_PATH}/src/MobilenetDetection.cpp             ${TKDNN_PATH}/src/Network.cpp
    ${TKDNN_PATH}/src/MulAdd.cpp                         ${TKDNN_PATH}/src/NetworkRT.cpp
    ${TKDNN_PATH}/src/NetworkViz.cpp                     ${TKDNN_PATH}/src/Pooling.cpp
    ${TKDNN_PATH}/src/Region.cpp                         ${TKDNN_PATH}/src/Route.cpp
    ${TKDNN_PATH}/src/Reorg.cpp                          ${TKDNN_PATH}/src/Shortcut.cpp
    ${TKDNN_PATH}/src/Reshape.cpp                        ${TKDNN_PATH}/src/Softmax.cpp
    ${TKDNN_PATH}/src/Upsample.cpp                       ${TKDNN_PATH}/src/utils.cpp
    ${TKDNN_PATH}/src/Yolo.cpp                           ${TKDNN_PATH}/src/Yolo3Detection.cpp
) 
set(tkdnn_LIBS kernels ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)
include_directories(${TKDNN_PATH}/include ${CUDA_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS} ${NVINFER_INCLUDES})
add_library(tkDNN SHARED ${TKDNN_CORE_FILES})
target_link_libraries(tkDNN ${tkdnn_LIBS})

# DARKNET
set(DARKNET_SRC_FILES
    ${TKDNN_PATH}/tests/darknet/csresnext50-panet-spp_berkeley.cpp                     ${TKDNN_PATH}/tests/darknet/csresnext50-panet-spp.cpp
    ${TKDNN_PATH}/tests/darknet/viz_yolo3.cpp                                          ${TKDNN_PATH}/tests/darknet/yolo2tiny.cpp
    ${TKDNN_PATH}/tests/darknet/yolo2_voc.cpp                                          ${TKDNN_PATH}/tests/darknet/yolo3_512.cpp
    ${TKDNN_PATH}/tests/darknet/yolo2.cpp                                              ${TKDNN_PATH}/tests/darknet/yolo3_berkeley.cpp
    ${TKDNN_PATH}/tests/darknet/yolo3.cpp                                              ${TKDNN_PATH}/tests/darknet/yolo3_coco4.cpp
    ${TKDNN_PATH}/tests/darknet/yolo3tiny_512.cpp                                      ${TKDNN_PATH}/tests/darknet/yolo3_flir.cpp
    ${TKDNN_PATH}/tests/darknet/yolo3tiny.cpp                                          ${TKDNN_PATH}/tests/darknet/yolo4_mmr.cpp
    ${TKDNN_PATH}/tests/darknet/yolo4_berkeley.cpp                                     ${TKDNN_PATH}/tests/darknet/yolo4.cpp
    ${TKDNN_PATH}/tests/darknet/yolo4tiny.cpp                                          ${TKDNN_PATH}/tests/darknet/yolo4x.cpp
)
foreach(test_SRC ${DARKNET_SRC_FILES})
    get_filename_component(test_NAME "${test_SRC}" NAME_WE)
    set(test_NAME test_${test_NAME})
    add_executable(${test_NAME} ${test_SRC})
    target_link_libraries(${test_NAME} tkDNN)
endforeach()

set(tkDNN_LIBRARIES 
    tkDNN 
    kernels 
    ${CUDA_LIBRARIES} 
    ${CUDA_CUBLAS_LIBRARIES}
	  ${CUDNN_LIBRARIES}
	  ${OpenCV_LIBS}
)

set(PROJECT_LIB_FILES
    src/tkDNNObjectDetector.cpp 
)

link_directories(
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

cuda_add_library(${PROJECT_NAME}_lib
    ${TKDNN_CUDA_FILES}
    ${TKDNN_CORE_FILES}
    ${PROJECT_LIB_FILES}
)

target_link_libraries(${PROJECT_NAME}_lib
    cuda
    cudart
    cublas
    curand
    ${tkDNN_LIBRARIES}
)

cuda_add_executable(${PROJECT_NAME}
    src/tkDNNobjectdetector_node.cpp
)

target_link_libraries(${PROJECT_NAME}_lib
  m
  pthread
  stdc++
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

target_link_libraries(${PROJECT_NAME}
  ${PROJECT_NAME}_lib
)

add_dependencies(${PROJECT_NAME}_lib
  darknet_ros_msgs_generate_messages_cpp
)

install(TARGETS ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY config launch yolo_network_config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#########################
###   CLANG TOOLING   ###
#########################
find_package(cmake_clang_tools QUIET)
if (cmake_clang_tools_FOUND)
  message(STATUS "Run clang tooling")
  add_clang_tooling(
    TARGETS ${PROJECT_NAME}
    SOURCE_DIRS ${CMAKE_CURRENT_LIST_DIR}/src ${CMAKE_CURRENT_LIST_DIR}/include ${CMAKE_CURRENT_LIST_DIR}/test
    CT_HEADER_DIRS ${CMAKE_CURRENT_LIST_DIR}/include
    CF_WERROR
  )
endif (cmake_clang_tools_FOUND)
