cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(runtime)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Protobuf REQUIRED)
if(NOT TARGET spdlog)
    # Stand-alone build
    find_package(spdlog REQUIRED)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread -g")
set(CMAKE_BUILD_PARALLEL_LEVEL 4)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(../../include)
include_directories(../../proto)

file(GLOB ALL_SOURCES ${PROJECT_SOURCE_DIR}/*.cpp ../../src/*.cpp ../../proto/*.cc)
# file(GLOB ALL_SOURCES ${PROJECT_SOURCE_DIR}/*.cpp ../../proto/*.cc)
add_executable(runtime ${ALL_SOURCES})

target_link_libraries(runtime ${TORCH_LIBRARIES})
target_link_libraries(runtime ${OpenCV_LIBS})
target_link_libraries(runtime ${Protobuf_LIBRARIES})
target_link_libraries(runtime stdc++fs)
target_link_libraries(runtime spdlog::spdlog)

set_property(TARGET runtime PROPERTY CXX_STANDARD 17)


# Incremental Build
if(CMAKE_GENERATOR STREQUAL "Ninja")
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-fdiagnostics-generate-patch" SUPPORTS_DIAGNOSTICS_GENERATE_PATCH)
    if(SUPPORTS_DIAGNOSTICS_GENERATE_PATCH)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-generate-patch")
    endif()
endif()