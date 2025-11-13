cmake_minimum_required(VERSION 3.1)
project(mono_vo)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4 REQUIRED)

add_executable(mono_vo src/main.cpp)

target_link_libraries( mono_vo ${OpenCV_LIBS} )