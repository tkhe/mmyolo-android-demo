project(mmyolo)

cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.5.4-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20220420-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(mmyolo SHARED mmyolo.cpp basenet.cpp rtmdet.cpp yolov5.cpp ndkcamera.cpp)

target_link_libraries(mmyolo ncnn ${OpenCV_LIBS} camera2ndk mediandk)
