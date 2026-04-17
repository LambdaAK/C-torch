#include "math/matrix.hpp"
#include "math/ast.hpp"

namespace ml
{

    enum class KernelType
    {
        Linear,
        Polynomial_2,
        Polynomial_3,
        Radial_Basis
    };

    class KernelOptions
    {
    private:
        KernelType kernel_type;
        double gamma;

    public:
        KernelOptions(KernelType kernel_type = KernelType::Linear, double gamma = 1.0)
            : kernel_type(kernel_type), gamma(gamma) {}

        static KernelOptions linear()
        {
            return KernelOptions(KernelType::Linear);
        }

        static KernelOptions polynomial_2()
        {
            return KernelOptions(KernelType::Polynomial_2);
        }

        static KernelOptions polynomial_3()
        {
            return KernelOptions(KernelType::Polynomial_3);
        }

        static KernelOptions radial_basis(double gamma)
        {
            return KernelOptions(KernelType::Radial_Basis, gamma);
        }

        KernelType get_kernel_type() const
        {
            return kernel_type;
        }

        double get_gamma() const
        {
            return gamma;
        }
    };

    class Kernels
    {
    public:
        static double linear(const Matrix &x1, const Matrix &x2)
        {
            return x1.inner_product(x2);
        }

        static double polynomial_2(const Matrix &x1, const Matrix &x2)
        {
            return std::pow(1 + x1.inner_product(x2), 2);
        }

        static double polynomial_3(const Matrix &x1, const Matrix &x2)
        {
            return std::pow(1 + x1.inner_product(x2), 3);
        }

        /** Standard RBF (Gaussian) kernel: exp(-gamma * ||x1 - x2||^2). */
        static double radial_basis(const Matrix &x1, const Matrix &x2, double gamma = 1.0)
        {
            const double d = x1.euclideanDistance(x2);
            return std::exp(-gamma * d * d);
        }

        static double kernel(const Matrix &x, const Matrix &y, KernelOptions kernel_options)
        {
            switch (kernel_options.get_kernel_type())
            {
            case KernelType::Linear:
                return linear(x, y);
            case KernelType::Polynomial_2:
                return polynomial_2(x, y);
            case KernelType::Polynomial_3:
                return polynomial_3(x, y);
            case KernelType::Radial_Basis:
                return radial_basis(x, y, kernel_options.get_gamma());
            default:
                throw std::invalid_argument("Invalid kernel type");
            }
        }
    };

    class KernelSVM
    {
    private:
        Matrix xTr;
        Matrix yTr;
        std::vector<Matrix> xTr_rows;
        Matrix weights;
        double bias;
        double learning_rate;
        int max_iter;
        double C;
        KernelOptions kernel_options;

    public:
        KernelSVM(Matrix xTr, Matrix yTr, double learning_rate, int max_iter, double C, KernelOptions kernel_options);
        std::shared_ptr<math::ASTNode> loss_function();
        int predict(const Matrix &x) const;
        double score(const Matrix &xTe, const Matrix &yTe) const;
    };
}