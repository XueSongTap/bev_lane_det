cmake_minimum_required(VERSION 3.22)
project(test)

set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3")
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
message(${OpenCV_INCLUDE_DIRS})



#target_include_directories(<target> PUBLIC
#        ${OpenCV_INCLUDE_DIRS}
#         )
add_executable(test main.cpp)
target_link_libraries(test Eigen3::Eigen ${OpenCV_LIBS})