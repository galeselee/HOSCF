cmake_minimum_required(VERSION 3.9)
project(run_als)

set(SOURCE als.cpp)
SET(CMAKE_BUILD_TYPE "Relese")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -Ofast -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -Ofast -Wall")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -Ofast -Wall")
set(cxx17 "-std=c++17")

include_directories(../include)

add_executable(run_als ${SOURCE})
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

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LAPACKLINKER_FLAGS}")
