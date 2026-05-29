cmake_minimum_required(VERSION 3.4.1)
project(MyLib)

set (CMAKE_VERBOSE_MAKEFILE ON)
set (CMAKE_CXX_STANDARD 17)

add_library(react-native-my-lib SHARED ../cpp/MyLib.cpp cpp-adapter.cpp)

find_package(ReactAndroid REQUIRED CONFIG)

include_directories(../cpp)

target_link_libraries(react-native-my-lib ReactAndroid::jsi)
