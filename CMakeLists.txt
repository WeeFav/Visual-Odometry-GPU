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

add_executable(orb 
    src/orb.cpp
    src/GaussianBlur.cu
    src/Fast.cu
    src/Orientations.cu
    src/Brief.cu
)
target_link_libraries(orb ${OpenCV_LIBS})
target_include_directories(orb PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

add_executable(fast 
    src/fast.cpp
    src/Fast.cu
)
target_link_libraries(fast ${OpenCV_LIBS})
target_include_directories(fast PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
