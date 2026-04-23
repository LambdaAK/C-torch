#include "nn_optim.hpp"
#include "math/parallel.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <mutex>
#include <string>
#include <utility>
#include <type_traits>

namespace ml
{
namespace
{
constexpr std::uint64_t kOptimizerStateMagic = 0x4354524F50535431ULL; // "CTORPST1"

template <typename T>
void append_pod(std::string &buffer, const T &value)
{
    static_assert(std::is_trivially_copyable_v<T>, "append_pod requires a trivially copyable type.");
    const char *bytes = reinterpret_cast<const char *>(&value);
    buffer.append(bytes, sizeof(T));
}

template <typename T>
T read_pod(const std::string &buffer, std::size_t &offset)
{
    static_assert(std::is_trivially_copyable_v<T>, "read_pod requires a trivially copyable type.");
    if (offset + sizeof(T) > buffer.size())
    {
        throw std::runtime_error("Corrupt optimizer state blob.");
    }

    T value{};
    std::memcpy(&value, buffer.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

void append_matrix(std::string &buffer, const Matrix &matrix)
{
    append_pod<std::uint64_t>(buffer, static_cast<std::uint64_t>(matrix.numRows()));
    append_pod<std::uint64_t>(buffer, static_cast<std::uint64_t>(matrix.numCols()));
    for (std::size_t row = 0; row < matrix.numRows(); ++row)
    {
        for (std::size_t col = 0; col < matrix.numCols(); ++col)
        {
            append_pod<double>(buffer, matrix(row, col));
        }
    }
}

Matrix read_matrix(const std::string &buffer, std::size_t &offset)
{
    const std::uint64_t rows = read_pod<std::uint64_t>(buffer, offset);
    const std::uint64_t cols = read_pod<std::uint64_t>(buffer, offset);
    Matrix matrix(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
    for (std::size_t row = 0; row < matrix.numRows(); ++row)
    {
        for (std::size_t col = 0; col < matrix.numCols(); ++col)
        {
            matrix(row, col) = read_pod<double>(buffer, offset);
        }
    }
    return matrix;
}
} // namespace

    NNOptimizer::NNOptimizer(
        std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> p,
        float lr,
        size_t batch_size,
        NNOptimType optim_type,
        float beta1,
        float beta2,
        float epsilon,
        float rho,
        float weight_decay)
        : lr(lr),
          batch_size(batch_size),
          optim_type(optim_type),
          beta1(beta1),
          beta2(beta2),
          epsilon(epsilon),
          rho(rho),
          weight_decay(weight_decay),
          step_count(0),
          params(std::move(p))
    {
        first_moment.reserve(params.size());
        second_moment.reserve(params.size());
        grad_accumulator.reserve(params.size());

        for (const auto &p : params)
        {
            if (!p.first || !p.second)
            {
                throw std::invalid_argument("NNOptimizer received null parameter or gradient pointer.");
            }

            const size_t rows = p.first->numRows();
            const size_t cols = p.first->numCols();

            if (p.second->numRows() != rows || p.second->numCols() != cols)
            {
                throw std::invalid_argument("NNOptimizer parameter and gradient dimensions do not match.");
            }

            first_moment.emplace_back(rows, cols);
            second_moment.emplace_back(rows, cols);
            grad_accumulator.emplace_back(rows, cols);
        }
    }

    std::string NNOptimizer::serialize_state() const
    {
        std::string buffer;
        append_pod<std::uint64_t>(buffer, kOptimizerStateMagic);
        append_pod<std::uint64_t>(buffer, 1ULL); // version
        append_pod<std::uint64_t>(buffer, static_cast<std::uint64_t>(params.size()));
        append_pod<std::uint64_t>(buffer, static_cast<std::uint64_t>(batch_size));
        append_pod<std::uint64_t>(buffer, static_cast<std::uint64_t>(step_count));
        append_pod<std::uint64_t>(buffer, static_cast<std::uint64_t>(optim_type));
        append_pod<float>(buffer, lr);
        append_pod<float>(buffer, beta1);
        append_pod<float>(buffer, beta2);
        append_pod<float>(buffer, epsilon);
        append_pod<float>(buffer, rho);
        append_pod<float>(buffer, weight_decay);

        for (std::size_t index = 0; index < params.size(); ++index)
        {
            append_matrix(buffer, first_moment[index]);
            append_matrix(buffer, second_moment[index]);
            append_matrix(buffer, grad_accumulator[index]);
        }

        return buffer;
    }

    void NNOptimizer::deserialize_state(const std::string &blob)
    {
        std::size_t offset = 0;
        const std::uint64_t magic = read_pod<std::uint64_t>(blob, offset);
        if (magic != kOptimizerStateMagic)
        {
            throw std::runtime_error("Invalid optimizer state magic.");
        }

        const std::uint64_t version = read_pod<std::uint64_t>(blob, offset);
        if (version != 1ULL)
        {
            throw std::runtime_error("Unsupported optimizer state version.");
        }

        const std::uint64_t param_count = read_pod<std::uint64_t>(blob, offset);
        const std::uint64_t saved_batch_size = read_pod<std::uint64_t>(blob, offset);
        const std::uint64_t saved_step_count = read_pod<std::uint64_t>(blob, offset);
        const std::uint64_t saved_type = read_pod<std::uint64_t>(blob, offset);
        const float saved_lr = read_pod<float>(blob, offset);
        const float saved_beta1 = read_pod<float>(blob, offset);
        const float saved_beta2 = read_pod<float>(blob, offset);
        const float saved_epsilon = read_pod<float>(blob, offset);
        const float saved_rho = read_pod<float>(blob, offset);
        const float saved_weight_decay = read_pod<float>(blob, offset);

        if (param_count != params.size())
        {
            throw std::runtime_error("Optimizer state parameter count mismatch.");
        }

        batch_size = static_cast<std::size_t>(saved_batch_size);
        step_count = static_cast<std::size_t>(saved_step_count);
        optim_type = static_cast<NNOptimType>(saved_type);
        lr = saved_lr;
        beta1 = saved_beta1;
        beta2 = saved_beta2;
        epsilon = saved_epsilon;
        rho = saved_rho;
        weight_decay = saved_weight_decay;

        first_moment.clear();
        second_moment.clear();
        grad_accumulator.clear();
        first_moment.reserve(params.size());
        second_moment.reserve(params.size());
        grad_accumulator.reserve(params.size());

        for (std::size_t index = 0; index < params.size(); ++index)
        {
            Matrix m = read_matrix(blob, offset);
            Matrix v = read_matrix(blob, offset);
            Matrix acc = read_matrix(blob, offset);

            const size_t rows = params[index].first->numRows();
            const size_t cols = params[index].first->numCols();
            if (m.numRows() != rows || m.numCols() != cols ||
                v.numRows() != rows || v.numCols() != cols ||
                acc.numRows() != rows || acc.numCols() != cols)
            {
                throw std::runtime_error("Optimizer state shape mismatch.");
            }

            first_moment.push_back(std::move(m));
            second_moment.push_back(std::move(v));
            grad_accumulator.push_back(std::move(acc));
        }

        if (offset != blob.size())
        {
            throw std::runtime_error("Trailing bytes detected in optimizer state blob.");
        }
    }

    bool NNOptimizer::save_state(const std::string &filepath) const
    {
        std::ofstream file(filepath, std::ios::binary | std::ios::trunc);
        if (!file.is_open())
        {
            return false;
        }

        const std::string blob = serialize_state();
        file.write(blob.data(), static_cast<std::streamsize>(blob.size()));
        return file.good();
    }

    bool NNOptimizer::load_state(const std::string &filepath)
    {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open())
        {
            return false;
        }

        const std::string blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        try
        {
            deserialize_state(blob);
        }
        catch (...)
        {
            return false;
        }

        return true;
    }

    void NNOptimizer::zero_grad()
    {
        ctorch::parallel::parallel_for_items(
            params.size(),
            1,
            [&](std::size_t begin, std::size_t end)
            {
                for (std::size_t idx = begin; idx < end; ++idx)
                {
                    (*params[idx].second) = 0.0 * (*params[idx].second); // zero the gradients
                }
            },
            2);
    }

    void NNOptimizer::step()
    {
        if (batch_size == 0)
        {
            throw std::invalid_argument("NNOptimizer::step called with batch_size = 0.");
        }

        ++step_count;
        const float inv_batch_size = 1.0f / static_cast<float>(batch_size);
        const float one_minus_beta1_t = 1.0f - std::pow(beta1, static_cast<float>(step_count));
        const float one_minus_beta2_t = 1.0f - std::pow(beta2, static_cast<float>(step_count));

        ctorch::parallel::parallel_for_items(
            params.size(),
            1,
            [&](std::size_t begin, std::size_t end)
            {
                for (std::size_t idx = begin; idx < end; ++idx)
                {
                    Matrix &param = *params[idx].first;
                    const Matrix grad = inv_batch_size * (*params[idx].second);
                    Matrix &m = first_moment[idx];
                    Matrix &v = second_moment[idx];
                    Matrix &acc = grad_accumulator[idx];

                    for (size_t i = 0; i < param.numRows(); ++i)
                    {
                        for (size_t j = 0; j < param.numCols(); ++j)
                        {
                            const float g = static_cast<float>(grad(i, j));

                            if (optim_type == NNOptimType::SGD)
                            {
                                param(i, j) -= lr * g;
                                continue;
                            }

                            if (optim_type == NNOptimType::ADAGRAD)
                            {
                                acc(i, j) = acc(i, j) + g * g;
                                param(i, j) -= lr * g / (std::sqrt(acc(i, j)) + epsilon);
                                continue;
                            }

                            if (optim_type == NNOptimType::RMSPROP)
                            {
                                v(i, j) = rho * v(i, j) + (1.0f - rho) * g * g;
                                param(i, j) -= lr * g / (std::sqrt(v(i, j)) + epsilon);
                                continue;
                            }

                            if (optim_type == NNOptimType::ADAM || optim_type == NNOptimType::ADAMW)
                            {
                                if (optim_type == NNOptimType::ADAMW)
                                {
                                    // Decoupled weight decay.
                                    param(i, j) -= lr * weight_decay * static_cast<float>(param(i, j));
                                }

                                m(i, j) = beta1 * m(i, j) + (1.0f - beta1) * g;
                                v(i, j) = beta2 * v(i, j) + (1.0f - beta2) * g * g;
                                const float m_hat = m(i, j) / one_minus_beta1_t;
                                const float v_hat = v(i, j) / one_minus_beta2_t;
                                param(i, j) -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
                                continue;
                            }

                            throw std::runtime_error("Unknown NN optimizer type.");
                        }
                    }
                }
            },
            2);
    }
} // namespace ml
