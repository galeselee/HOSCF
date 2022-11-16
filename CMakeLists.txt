cmake_minimum_required(VERSION 3.9)
project(rank-one)

set(SOURCE rank-one.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(cxx17 "-std=c++17")

add_executable(rank-one ${SOURCE})
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("${cxx17}" COMPILER_SUPPORTS_CXX17)
if(COMPILER_SUPPORTS_CXX17)
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has C++17 support, use C++17.")
    set(CMAKE_CXX_STANDARD 17)
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif(COMPILER_SUPPORTS_CXX17)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC ${BLAS_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${LAPACK_LIBRARIES})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LAPACKLINKER_FLAGS}")
