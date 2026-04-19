add_executable(kernel_circle_demo main.cpp)

target_include_directories(kernel_circle_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/lib
)
target_link_libraries(kernel_circle_demo PRIVATE ctorch)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(kernel_circle_demo PRIVATE -Wall -Wextra)
endif()

set_target_properties(kernel_circle_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/demos/kernel
)
