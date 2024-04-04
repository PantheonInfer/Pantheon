cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(profiler)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB ALL_SOURCES ${PROJECT_SOURCE_DIR}/*.cpp)
add_executable(profiler ${ALL_SOURCES})

target_link_libraries(profiler ${TORCH_LIBRARIES})
target_link_libraries(profiler ${OpenCV_LIBS})

set_property(TARGET profiler PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
