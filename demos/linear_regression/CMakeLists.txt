add_executable(linear_regression_demo main.cpp)

target_include_directories(linear_regression_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/lib
)
target_link_libraries(linear_regression_demo PRIVATE ctorch)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(linear_regression_demo PRIVATE -Wall -Wextra)
endif()

set_target_properties(linear_regression_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/demos/linear_regression
)
