cmake_minimum_required(VERSION 3.9)
project(run)

set(SOURCE main.cpp)

#if want gdb
#SET(CMAKE_BUILD_TYPE Debug)

SET(CMAKE_BUILD_TYPE "Relese")
SET(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -Ofast -Wall")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -Ofast -Wall")
SET(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(include)
add_subdirectory(src)


add_executable(${PROJECT_NAME} ${SOURCE})
target_link_libraries(${PROJECT_NAME} ${MPI_C_LIBRARIES})
target_link_libraries(${PROJECT_NAME} src_lib)
