project(src_lib)

add_subdirectory(ttvc)

include_directories(../include)
include_directories(ttvc/include)
file(GLOB SRC *.cpp)

add_library(${PROJECT_NAME} SHARED ${SRC})

target_link_libraries(${PROJECT_NAME} PUBLIC ttvc_lib)

SET(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -Ofast -Wall ${CMAKE_CXX_FLAGS}")
set(cxx17 "-std=c++17")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("${cxx17}" COMPILER_SUPPORTS_CXX17)
if(COMPILER_SUPPORTS_CXX17)
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has C++17 support, use C++17.")
    set(CMAKE_CXX_STANDARD 17)
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif(COMPILER_SUPPORTS_CXX17)

set(BLA_VENDOR Intel10_64lp_seq)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC ${BLAS_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${LAPACK_LIBRARIES})
