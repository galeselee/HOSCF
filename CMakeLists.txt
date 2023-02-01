cmake_minimum_required(VERSION 3.9)
project(run)

set(SOURCE main.cpp)
SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_STANDARD 17)

include_directories(include)
add_subdirectory(src)

add_executable(${PROJECT_NAME} ${SOURCE})

target_link_libraries(${PROJECT_NAME} src_lib)

