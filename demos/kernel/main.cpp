#include "math/matrix.hpp"
#include "ml/kernelsvm.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace
{
    struct Sample
    {
        double x;
        double y;
        int label;
    };

    class CoutSilencer
    {
    private:
        std::ostringstream sink;
        std::streambuf *old_buf;

    public:
        CoutSilencer()
            : old_buf(std::cout.rdbuf(sink.rdbuf()))
        {
        }

        CoutSilencer(const CoutSilencer &) = delete;
        CoutSilencer &operator=(const CoutSilencer &) = delete;

        ~CoutSilencer()
        {
            std::cout.rdbuf(old_buf);
        }
    };

    std::vector<Sample> generate_circle_samples(
        size_t count,
        double radius,
        double radius_noise,
        int label,
        std::mt19937 &rng)
    {
        std::vector<Sample> samples;
        samples.reserve(count);

        std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * std::numbers::pi_v<double>);
        std::normal_distribution<double> noise_dist(0.0, radius_noise);

        for (size_t i = 0; i < count; ++i)
        {
            const double theta = angle_dist(rng);
            const double r = radius + noise_dist(rng);
            samples.push_back(Sample{
                r * std::cos(theta),
                r * std::sin(theta),
                label,
            });
        }

        return samples;
    }

    Matrix samples_to_features(const std::vector<Sample> &samples)
    {
        Matrix x(samples.size(), 2);
        for (size_t i = 0; i < samples.size(); ++i)
        {
            x(i, 0) = samples[i].x;
            x(i, 1) = samples[i].y;
        }
        return x;
    }

    Matrix samples_to_labels(const std::vector<Sample> &samples)
    {
        Matrix y(1, samples.size());
        for (size_t i = 0; i < samples.size(); ++i)
        {
            y(0, i) = samples[i].label;
        }
        return y;
    }

    Matrix sample_to_row_matrix(const Sample &sample)
    {
        Matrix row(1, 2);
        row(0, 0) = sample.x;
        row(0, 1) = sample.y;
        return row;
    }
}

int main()
{
    try
    {
        constexpr size_t kSamplesPerClass = 8;
        constexpr size_t kTrainPerClass = 6;
        constexpr double kInnerRadius = 1.0;
        constexpr double kOuterRadius = 2.8;
        constexpr double kRadiusNoise = 0.02;
        constexpr double kLearningRate = 0.01;
        constexpr int kMaxIter = 100;
        constexpr double kPenalty = 1.0;
        constexpr double kGamma = 1.5;

        std::mt19937 rng(7);
        const std::vector<Sample> inner = generate_circle_samples(kSamplesPerClass, kInnerRadius, kRadiusNoise, -1, rng);
        const std::vector<Sample> outer = generate_circle_samples(kSamplesPerClass, kOuterRadius, kRadiusNoise, 1, rng);

        std::vector<Sample> all_samples;
        all_samples.reserve(2 * kSamplesPerClass);
        all_samples.insert(all_samples.end(), inner.begin(), inner.end());
        all_samples.insert(all_samples.end(), outer.begin(), outer.end());
        std::shuffle(all_samples.begin(), all_samples.end(), rng);

        std::vector<Sample> train_samples(all_samples.begin(), all_samples.begin() + static_cast<std::ptrdiff_t>(2 * kTrainPerClass));
        std::vector<Sample> test_samples(all_samples.begin() + static_cast<std::ptrdiff_t>(2 * kTrainPerClass), all_samples.end());

        const Matrix x_tr = samples_to_features(train_samples);
        const Matrix y_tr = samples_to_labels(train_samples);
        const Matrix x_te = samples_to_features(test_samples);
        const Matrix y_te = samples_to_labels(test_samples);

        std::unique_ptr<ml::KernelSVM> model;
        {
            CoutSilencer silence;
            model = std::make_unique<ml::KernelSVM>(
                x_tr,
                y_tr,
                kLearningRate,
                kMaxIter,
                kPenalty,
                ml::KernelOptions::radial_basis(kGamma));
        }

        const double train_accuracy = model->score(x_tr, y_tr);
        const double test_accuracy = model->score(x_te, y_te);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Kernel SVM\n";
        std::cout << "Concentric circles demo\n";
        std::cout << "Class -1: inner circle, radius " << kInnerRadius << "\n";
        std::cout << "Class +1: outer circle, radius " << kOuterRadius << "\n";
        std::cout << "Training samples: " << train_samples.size() << "\n";
        std::cout << "Test samples: " << test_samples.size() << "\n";
        std::cout << "Kernel: radial basis (gamma=" << kGamma << ")\n";
        std::cout << "Train accuracy: " << train_accuracy << "\n";
        std::cout << "Test accuracy: " << test_accuracy << "\n\n";

        std::cout << "Test predictions:\n";
        for (const Sample &sample : test_samples)
        {
            const Matrix row = sample_to_row_matrix(sample);
            const int prediction = model->predict(row);
            std::cout << "  (" << sample.x << ", " << sample.y << ")"
                      << " label=" << sample.label
                      << " prediction=" << prediction << "\n";
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Kernel demo failed: " << e.what() << '\n';
        return 1;
    }
}
