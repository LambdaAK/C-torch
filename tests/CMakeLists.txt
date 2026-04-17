if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_executable(ctorch_tests
    test_matrix.cpp
    test_knn.cpp
    test_mab.cpp
    test_ucb.cpp
    test_pca.cpp
    test_dqn.cpp
)

target_link_libraries(ctorch_tests PRIVATE ctorch GTest::gtest_main)

target_include_directories(ctorch_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/lib
    ${CMAKE_SOURCE_DIR}/experiments/ndtictactoe
)

target_sources(ctorch_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/experiments/ndtictactoe/dqn.cpp
    ${CMAKE_SOURCE_DIR}/experiments/ndtictactoe/replaymemory.cpp
    ${CMAKE_SOURCE_DIR}/experiments/ndtictactoe/sample.cpp
)

include(GoogleTest)
gtest_discover_tests(ctorch_tests)
