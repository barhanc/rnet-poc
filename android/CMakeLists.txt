cmake_minimum_required(VERSION 3.10)
project(MyLib)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

set(EXEC_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../third-party/android/jniLibs/${ANDROID_ABI}/libexecutorch.so")

add_library(executorch SHARED IMPORTED)
set_target_properties(executorch PROPERTIES IMPORTED_LOCATION ${EXEC_LIB_PATH})

find_package(ReactAndroid REQUIRED CONFIG)

add_library(${CMAKE_PROJECT_NAME} SHARED 
    ../cpp/MyLib.cpp 
    cpp-adapter.cpp
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE 
    ../cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../third-party
    ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/executorch/runtime/core/portable_type/c10
    ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/executorch/.venv/lib/python3.12/site-packages/torch/include
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    executorch
    ReactAndroid::jsi  # Use the namespaced target from find_package
    log
)