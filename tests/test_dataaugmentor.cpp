#include <gtest/gtest.h>
#include "math/dataaugmentor.hpp"
#include "math/matrix.hpp"

TEST(DataAugmentor, RandomFourierRejectsInvalidParams) {
    Matrix x(2, 2);
    EXPECT_THROW(DataAugmentor::random_fourier_features(x, 0, 0.5), std::invalid_argument);
    EXPECT_THROW(DataAugmentor::random_fourier_features(x, -1, 0.5), std::invalid_argument);
    EXPECT_THROW(DataAugmentor::random_fourier_features(x, 4, 0.0), std::invalid_argument);
    EXPECT_THROW(DataAugmentor::random_fourier_features(x, 4, -0.5), std::invalid_argument);
}
