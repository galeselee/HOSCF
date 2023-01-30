cmake_minimum_required(VERSION 3.16)
project(ttvc)

include_directories(./include)

add_subdirectory(src)
add_subdirectory(lib/googletest)
add_subdirectory(test)

file(GLOB SRC main.cpp)
add_executable(run ${SRC})