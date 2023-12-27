cmake_minimum_required(VERSION 3.10)
project(rt_nkf)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

# Find the PortAudio package
find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)

# Include the PortAudio headers
include_directories(${PORTAUDIO_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

# Create your executable
add_executable(rt_nkf main.cpp)

# Add subdirectories
add_subdirectory(modules)

# Set the location for the PortAudio library
target_link_directories(rt_nkf PUBLIC ${PORTAUDIO_LIBRARY_DIRS} /usr/local/lib)

target_link_libraries(rt_nkf PRIVATE ${PORTAUDIO_LIBRARIES} modules)
