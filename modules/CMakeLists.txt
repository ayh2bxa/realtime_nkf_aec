project(modules)
add_library(modules activations.h activations.cpp matrix_helper.h matrix_helper.cpp gru.h gru.cpp complex_gru.h complex_gru.cpp kgnet.h kgnet.cpp nkf.h nkf.cpp)

# Include directories for modules

target_include_directories(modules PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR} /usr/local/Cellar/libsndfile/1.2.0_1/include)
target_link_libraries(modules sndfile)