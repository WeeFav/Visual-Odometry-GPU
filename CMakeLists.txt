cmake_minimum_required(VERSION 3.1)
project(mono_vo CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4 REQUIRED)
find_package(Ceres REQUIRED)

add_executable(vo_matching src/feature_matching.cpp)
target_link_libraries(vo_matching ${OpenCV_LIBS})

add_executable(vo_tracking src/feature_tracking.cpp)
target_link_libraries(vo_tracking ${OpenCV_LIBS})

add_executable(vo_ba src/with_bundle_adjustment.cpp)
target_link_libraries(vo_ba ${OpenCV_LIBS} Ceres::ceres)

add_executable(compare 
    src/compare.cpp
    src/orb.cpp
    src/orb_cpu.cpp
    src/orb_pattern.cpp
    src/GaussianBlur.cpp
    src/cuda/GaussianBlur.cu
    src/cuda/Fast.cu
    src/cuda/Orientations.cu
    src/cuda/Brief.cu
    src/cuda/Convolution.cu
)
target_link_libraries(compare ${OpenCV_LIBS})
target_include_directories(compare PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
