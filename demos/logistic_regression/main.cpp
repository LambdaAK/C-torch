#include "math/matrix.hpp"
#include "math/optim.hpp"
#include "ml/logisticregression.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace
{
    struct ClassificationSample
    {
        double x1;
        double x2;
        int label;
    };

    std::vector<ClassificationSample> generate_blob_samples(
        size_t count,
        double center_x,
        double center_y,
        double spread,
        int label,
        std::mt19937 &rng)
    {
        std::vector<ClassificationSample> samples;
        samples.reserve(count);

        std::normal_distribution<double> x_dist(center_x, spread);
        std::normal_distribution<double> y_dist(center_y, spread);

        for (size_t i = 0; i < count; ++i)
        {
            samples.push_back(ClassificationSample{
                x_dist(rng),
                y_dist(rng),
                label,
            });
        }

        return samples;
    }

    Matrix samples_to_features(const std::vector<ClassificationSample> &samples)
    {
        Matrix x(samples.size(), 2);
        for (size_t i = 0; i < samples.size(); ++i)
        {
            x(i, 0) = samples[i].x1;
            x(i, 1) = samples[i].x2;
        }
        return x;
    }

    Matrix samples_to_labels(const std::vector<ClassificationSample> &samples)
    {
        Matrix y(1, samples.size());
        for (size_t i = 0; i < samples.size(); ++i)
        {
            y(0, i) = samples[i].label;
        }
        return y;
    }

    Matrix sample_to_row_matrix(const ClassificationSample &sample)
    {
        Matrix row(1, 2);
        row(0, 0) = sample.x1;
        row(0, 1) = sample.x2;
        return row;
    }
}

int main()
{
    try
    {
        constexpr size_t kSamplesPerClass = 8;
        constexpr size_t kTrainPerClass = 6;
        constexpr double kSpread = 0.25;
        constexpr double kLearningRate = 0.05;
        constexpr int kMaxIter = 120;

        std::mt19937 rng(7);
        std::vector<ClassificationSample> samples;
        const std::vector<ClassificationSample> class_zero = generate_blob_samples(kSamplesPerClass, -1.2, -1.0, kSpread, 0, rng);
        const std::vector<ClassificationSample> class_one = generate_blob_samples(kSamplesPerClass, 1.2, 1.0, kSpread, 1, rng);

        samples.reserve(2 * kSamplesPerClass);
        samples.insert(samples.end(), class_zero.begin(), class_zero.end());
        samples.insert(samples.end(), class_one.begin(), class_one.end());
        std::shuffle(samples.begin(), samples.end(), rng);

        std::vector<ClassificationSample> train_samples(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(2 * kTrainPerClass));
        std::vector<ClassificationSample> test_samples(samples.begin() + static_cast<std::ptrdiff_t>(2 * kTrainPerClass), samples.end());

        const Matrix x_tr = samples_to_features(train_samples);
        const Matrix y_tr = samples_to_labels(train_samples);
        const Matrix x_te = samples_to_features(test_samples);
        const Matrix y_te = samples_to_labels(test_samples);

        ml::LogisticRegression model(
            x_tr,
            y_tr,
            math::OptimParams(math::OptimType::GD, kLearningRate, kMaxIter),
            DataAugmentationType::NO_OP);

        const double train_accuracy = model.score(x_tr, y_tr);
        const double test_accuracy = model.score(x_te, y_te);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Logistic Regression demo\n";
        std::cout << "Class 0 center: (-1.2, -1.0)\n";
        std::cout << "Class 1 center: (1.2, 1.0)\n";
        std::cout << "Training samples: " << train_samples.size() << "\n";
        std::cout << "Test samples: " << test_samples.size() << "\n";
        std::cout << "Train accuracy: " << train_accuracy << "\n";
        std::cout << "Test accuracy: " << test_accuracy << "\n\n";

        std::cout << "Test predictions:\n";
        for (const ClassificationSample &sample : test_samples)
        {
            const int prediction = static_cast<int>(model.predict(sample_to_row_matrix(sample)));
            std::cout << "  (" << sample.x1 << ", " << sample.x2 << ")"
                      << " label=" << sample.label
                      << " prediction=" << prediction << "\n";
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Logistic regression demo failed: " << e.what() << '\n';
        return 1;
    }
}
