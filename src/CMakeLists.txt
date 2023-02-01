project(src_lib)

add_subdirectory(ttvc)

#include_directories(../include)
file(GLOB SRC *.cpp)

add_library(${PROJECT_NAME} SHARED ${SRC})

target_link_libraries(${PROJECT_NAME} PUBLIC ttvc_lib)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC ${BLAS_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${LAPACK_LIBRARIES})