# CMake entry point
cmake_minimum_required(VERSION 3.16)
project(thread-pool LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)

# Header only library
add_library(thread-pool INTERFACE)
target_include_directories(thread-pool INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(thread-pool INTERFACE Threads::Threads)

add_executable(main example/main.cpp)
target_link_libraries(main PRIVATE thread-pool)

enable_testing()
add_executable(test_thread_pool tests/test_thread_pool.cpp)
target_link_libraries(test_thread_pool PRIVATE thread-pool)
add_test(NAME test_thread_pool COMMAND test_thread_pool)
