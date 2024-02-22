project(ttvc_lib)

include_directories(../include)
file(GLOB SRC *.cpp)

add_library(ttvc_lib SHARED ${SRC})


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


find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

target_link_libraries(ttvc_lib PUBLIC OpenMP::OpenMP_CXX)
