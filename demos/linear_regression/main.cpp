#include "math/matrix.hpp"
#include "math/optim.hpp"
#include "ml/linearregression.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace
{
    struct RegressionSample
    {
        double x;
        double y;
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

    std::vector<RegressionSample> generate_samples(
        size_t count,
        double slope,
        double intercept,
        double noise_std,
        std::mt19937 &rng)
    {
        std::vector<RegressionSample> samples;
        samples.reserve(count);

        std::uniform_real_distribution<double> x_dist(-2.0, 2.0);
        std::normal_distribution<double> noise_dist(0.0, noise_std);

        for (size_t i = 0; i < count; ++i)
        {
            const double x = x_dist(rng);
            const double y = slope * x + intercept + noise_dist(rng);
            samples.push_back(RegressionSample{x, y});
        }

        return samples;
    }

    Matrix samples_to_features(const std::vector<RegressionSample> &samples)
    {
        Matrix x(samples.size(), 1);
        for (size_t i = 0; i < samples.size(); ++i)
        {
            x(i, 0) = samples[i].x;
        }
        return x;
    }

    Matrix samples_to_targets(const std::vector<RegressionSample> &samples)
    {
        Matrix y(1, samples.size());
        for (size_t i = 0; i < samples.size(); ++i)
        {
            y(0, i) = samples[i].y;
        }
        return y;
    }

    Matrix sample_to_row_matrix(double x)
    {
        Matrix row(1, 1);
        row(0, 0) = x;
        return row;
    }

    double mean_absolute_error(
        const ml::LinearRegression &model,
        const std::vector<RegressionSample> &samples)
    {
        double total_error = 0.0;
        for (const RegressionSample &sample : samples)
        {
            const double prediction = model.predict(sample_to_row_matrix(sample.x));
            total_error += std::abs(prediction - sample.y);
        }
        return total_error / static_cast<double>(samples.size());
    }
}

int main()
{
    try
    {
        constexpr size_t kTrainCount = 12;
        constexpr size_t kTestCount = 4;
        constexpr double kSlope = 2.5;
        constexpr double kIntercept = -0.7;
        constexpr double kNoiseStd = 0.03;
        constexpr double kLearningRate = 0.05;
        constexpr int kMaxIter = 200;

        std::mt19937 rng(11);
        const std::vector<RegressionSample> train_samples = generate_samples(kTrainCount, kSlope, kIntercept, kNoiseStd, rng);
        const std::vector<RegressionSample> test_samples = generate_samples(kTestCount, kSlope, kIntercept, kNoiseStd, rng);

        const Matrix x_tr = samples_to_features(train_samples);
        const Matrix y_tr = samples_to_targets(train_samples);
        std::unique_ptr<ml::LinearRegression> model;
        {
            CoutSilencer silence;
            model = std::make_unique<ml::LinearRegression>(x_tr, y_tr, kLearningRate, kMaxIter);
        }

        const double train_mae = mean_absolute_error(*model, train_samples);
        const double test_mae = mean_absolute_error(*model, test_samples);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Linear Regression demo\n";
        std::cout << "Target function: y = " << kSlope << "x + " << kIntercept << " + noise\n";
        std::cout << "Training samples: " << train_samples.size() << "\n";
        std::cout << "Test samples: " << test_samples.size() << "\n";
        std::cout << "Train MAE: " << train_mae << "\n";
        std::cout << "Test MAE: " << test_mae << "\n\n";

        std::cout << "Test predictions:\n";
        for (const RegressionSample &sample : test_samples)
        {
            const double prediction = model->predict(sample_to_row_matrix(sample.x));
            std::cout << "  x=" << sample.x
                      << " target=" << sample.y
                      << " prediction=" << prediction
                      << " abs_error=" << std::abs(prediction - sample.y) << "\n";
        }

        std::cout << "\nCheckpoints:\n";
        for (double x : {-1.50, -0.25, 0.75, 2.00})
        {
            const double target = kSlope * x + kIntercept;
            const double prediction = model->predict(sample_to_row_matrix(x));
            std::cout << "  x=" << x
                      << " target=" << target
                      << " prediction=" << prediction << "\n";
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Linear regression demo failed: " << e.what() << '\n';
        return 1;
    }
}
