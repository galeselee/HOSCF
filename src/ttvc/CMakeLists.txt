cmake_minimum_required(VERSION 3.16)
project(ttvc)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
include_directories(./include)

add_subdirectory(src)

#file(GLOB SRC main.cpp)
#add_executable(run ${SRC})

#target_link_libraries(run ttvc_lib)
