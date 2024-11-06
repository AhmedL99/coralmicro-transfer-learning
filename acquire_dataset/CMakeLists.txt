cmake_minimum_required(VERSION 3.13)

# Toolchain must be set before project() call.
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../coralmicro/cmake/toolchain-arm-none-eabi-gcc.cmake)
endif()

project(acquire-images)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../coralmicro)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../coralmicro ${CMAKE_BINARY_DIR}/coralmicro)

#add_executable_m7(acquire-images
#        main.cc
#)

# USB
add_executable_m7(acquire-images
    main.cc
    DATA
    ${CMAKE_CURRENT_SOURCE_DIR}/coral_micro_camera.html>/coral_micro_camera.html
)


target_link_libraries(
	acquire-images
        libs_base-m7_freertos
        libs_base-m7_http_server
        libs_jpeg_m7        
)
