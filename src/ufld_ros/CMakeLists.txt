cmake_minimum_required(VERSION 2.8)

project(ufld_ros)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL  "aarch64")
  set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
  set(CUDA_INSTALL_TARGET_DIR targets/aarch64-linux)
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(CMAKE_C_COMPILER /usr/bin/gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/g++)
  set(CUDA_INSTALL_TARGET_DIR targets/x86_64-linux)
endif()
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/include)

add_compile_options(-W)
add_compile_options(-std=c++11)

set( SMS 30 32 35 37 50 52 53 60 61 62 70 72 75)
foreach(sm ${SMS})
	set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
set(HIGHEST_SM 75)
set(GENCODE ${GENCODE} -gencode arch=compute_${HIGHEST_SM},code=compute_${HIGHEST_SM})
include_directories(${PROJECT_SOURCE_DIR}/include)
# 都没有include还写上？
find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  std_msgs
  message_generation
  image_transport
)

catkin_package(
#  INCLUDE_DIRS include
   CATKIN_DEPENDS roscpp sensor_msgs std_msgs message_runtime image_transport cv_bridge
)

include_directories(
    include/${PROJECT_NAME}
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
link_directories(${OpenCV_LIBS})
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/YOU_PATH/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1/TensorRT-7.2.3.4/include)
link_directories(/YOU_PATH/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1/TensorRT-7.2.3.4/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# cuda_add_library(myplugins SHARED include/${PROJECT_NAME}/yololayer.cu)
# target_link_libraries(myplugins nvinfer cudart)

# add_executable(yolov5_engine src/calibrator.cpp src/yolov5.cpp)
# target_link_libraries(yolov5_engine nvinfer cudart myplugins ${catkin_LIBRARIES} ${OpenCV_LIBS})

# add_executable(yolov5_gesture src/video_inteference_yolov5.cpp)
# target_link_libraries(yolov5_gesture nvinfer cudart myplugins ${catkin_LIBRARIES} ${OpenCV_LIBS})


set(CUDA_LIBS cudart cudnn cublas rt dl cudadevrt)
set(TRT_LIBS  nvparsers nvinfer nvonnxparser)
link_libraries(${CUDA_LIBS} ${TRT_LIBS} ${OpenCV_LIBS} ${catkin_LIBRARIES})
add_executable(lane_det 
				src/main.cpp
				src/logging.h
				src/utils.hpp
				)
target_link_libraries(lane_det ${CUDA_LIBS} ${OPENCV_LIBS} ${TRT_LIBS} ${catkin_LIBRARIES})


 
add_definitions(-O2 -pthread)
