cmake_minimum_required(VERSION 3.16)
project(ttvc)

#gdb
#set(CMAKE_BUILD_TYPE Debug)

SET(CMAKE_BUILD_TYPE "Relese")
SET(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -Ofast -Wall")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -Ofast -Wall")
set(CMAKE_CXX_STANDARD 17)
include_directories(./include)

add_subdirectory(src)

#file(GLOB SRC main.cpp)
#add_executable(run ${SRC})

#target_link_libraries(run ttvc_lib)
