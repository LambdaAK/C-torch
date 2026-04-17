cmake_minimum_required(VERSION 3.16)
project(ctorch CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CTORCH_SOURCES
    lib/math/matrix.cpp
    lib/math/dataaugmentor.cpp
    lib/ml/perceptron.cpp
    lib/ml/svm.cpp
    lib/ml/linearregression.cpp
    lib/ml/logisticregression.cpp
    lib/ml/knn.cpp
    lib/ml/nn.cpp
    lib/ml/randomfouriersvm.cpp
    lib/ml/kernelsvm.cpp
    lib/ml/kmeans.cpp
    lib/ml/mab.cpp
    lib/ml/ucb.cpp
    lib/ml/pca.cpp
    lib/ml/gaussian_nb.cpp
)

add_library(ctorch STATIC ${CTORCH_SOURCES})
target_include_directories(ctorch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib>
)

add_executable(classification experiments/classification/main.cpp)
target_include_directories(classification PRIVATE
    ${CMAKE_SOURCE_DIR}/lib
    ${CMAKE_SOURCE_DIR}/experiments/recommender
)
target_link_libraries(classification PRIVATE ctorch)

set(NDTICTACTOE_RL_SOURCES
    experiments/ndtictactoe/tictactoe.cpp
    experiments/ndtictactoe/replaymemory.cpp
    experiments/ndtictactoe/dqn.cpp
    experiments/ndtictactoe/reinforce.cpp
    experiments/ndtictactoe/sample.cpp
)

add_executable(tictactoe experiments/ndtictactoe/main.cpp ${NDTICTACTOE_RL_SOURCES})
target_include_directories(tictactoe PRIVATE
    ${CMAKE_SOURCE_DIR}/lib
    ${CMAKE_SOURCE_DIR}/experiments/ndtictactoe
)
target_link_libraries(tictactoe PRIVATE ctorch)

add_executable(ttt_main
    experiments/ndtictactoe/ttt_main.cpp
    experiments/ndtictactoe/tictactoe.cpp
    experiments/ndtictactoe/replaymemory.cpp
    experiments/ndtictactoe/dqn.cpp
    experiments/ndtictactoe/sample.cpp
)
target_include_directories(ttt_main PRIVATE
    ${CMAKE_SOURCE_DIR}/lib
    ${CMAKE_SOURCE_DIR}/experiments/ndtictactoe
)
target_link_libraries(ttt_main PRIVATE ctorch)

option(CTORCH_BUILD_TESTS "Build GoogleTest unit tests (downloads googletest)" ON)
include(CTest)
if(CTORCH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
