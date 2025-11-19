cmake_minimum_required(VERSION 3.1)
project(mono_vo)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4 REQUIRED)
find_package(Ceres REQUIRED)

add_executable(vo_matching src/feature_matching.cpp)
target_link_libraries(vo_matching ${OpenCV_LIBS})

add_executable(vo_ba src/with_bundle_adjustment.cpp)
target_link_libraries(vo_ba ${OpenCV_LIBS} Ceres::ceres)