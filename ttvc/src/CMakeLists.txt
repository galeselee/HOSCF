project(src)

include_directories(../include)
file(GLOB SRC *.cpp)

add_library(src_lib SHARED ${SRC})

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

target_link_libraries(src_lib PUBLIC OpenMP::OpenMP_CXX)
