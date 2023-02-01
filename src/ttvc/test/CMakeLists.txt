cmake_minimum_required(VERSION 3.16)
project(test)

include_directories(../include)


file(GLOB TEST_SRC *.cpp)
add_executable(${PROJECT_NAME} ${TEST_SRC})

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PUBLIC src_lib gtest)
