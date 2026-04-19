add_executable(logistic_regression_demo main.cpp)

target_include_directories(logistic_regression_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/lib
)
target_link_libraries(logistic_regression_demo PRIVATE ctorch)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(logistic_regression_demo PRIVATE -Wall -Wextra)
endif()

set_target_properties(logistic_regression_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/demos/logistic_regression
)
