#include "math/matrix.hpp"
#include "math/ast.hpp"

namespace ctorch::distributed
{
    class ProcessGroup;
}

namespace ml
{

    /**
     * @brief Supported kernel families for `KernelSVM`.
     */
    enum class KernelType
    {
        Linear,       ///< Linear dot-product kernel.
        Polynomial_2, ///< Degree-2 polynomial kernel.
        Polynomial_3, ///< Degree-3 polynomial kernel.
        Radial_Basis  ///< Gaussian radial basis function (RBF) kernel.
    };

    /**
     * @brief Immutable kernel configuration object.
     */
    class KernelOptions
    {
    private:
        KernelType kernel_type; ///< Selected kernel family.
        double gamma;           ///< RBF gamma (used only for `Radial_Basis`).

    public:
        /**
         * @brief Creates a kernel option bundle.
         * @param kernel_type Chosen kernel family.
         * @param gamma RBF gamma value when using radial basis kernel.
         */
        KernelOptions(KernelType kernel_type = KernelType::Linear, double gamma = 1.0)
            : kernel_type(kernel_type), gamma(gamma) {}

        /**
         * @brief Builds linear kernel options.
         * @return `KernelOptions` configured for linear kernel.
         */
        static KernelOptions linear()
        {
            return KernelOptions(KernelType::Linear);
        }

        /**
         * @brief Builds degree-2 polynomial kernel options.
         * @return `KernelOptions` configured for degree-2 polynomial kernel.
         */
        static KernelOptions polynomial_2()
        {
            return KernelOptions(KernelType::Polynomial_2);
        }

        /**
         * @brief Builds degree-3 polynomial kernel options.
         * @return `KernelOptions` configured for degree-3 polynomial kernel.
         */
        static KernelOptions polynomial_3()
        {
            return KernelOptions(KernelType::Polynomial_3);
        }

        /**
         * @brief Builds radial-basis kernel options.
         * @param gamma RBF gamma hyperparameter.
         * @return `KernelOptions` configured for RBF kernel.
         */
        static KernelOptions radial_basis(double gamma)
        {
            return KernelOptions(KernelType::Radial_Basis, gamma);
        }

        /**
         * @brief Returns the selected kernel family.
         * @return Kernel type enum.
         */
        KernelType get_kernel_type() const
        {
            return kernel_type;
        }

        /**
         * @brief Returns configured gamma value.
         * @return RBF gamma hyperparameter.
         */
        double get_gamma() const
        {
            return gamma;
        }
    };

    /**
     * @brief Stateless kernel function implementations.
     */
    class Kernels
    {
    public:
        /**
         * @brief Computes linear kernel.
         * @param x1 First sample row vector.
         * @param x2 Second sample row vector.
         * @return Dot product `x1 · x2`.
         */
        static double linear(const Matrix &x1, const Matrix &x2)
        {
            return x1.inner_product(x2);
        }

        /**
         * @brief Computes degree-2 polynomial kernel.
         * @param x1 First sample row vector.
         * @param x2 Second sample row vector.
         * @return `(1 + x1 · x2)^2`.
         */
        static double polynomial_2(const Matrix &x1, const Matrix &x2)
        {
            return std::pow(1 + x1.inner_product(x2), 2);
        }

        /**
         * @brief Computes degree-3 polynomial kernel.
         * @param x1 First sample row vector.
         * @param x2 Second sample row vector.
         * @return `(1 + x1 · x2)^3`.
         */
        static double polynomial_3(const Matrix &x1, const Matrix &x2)
        {
            return std::pow(1 + x1.inner_product(x2), 3);
        }

        /**
         * @brief Computes Gaussian radial basis function kernel.
         * @param x1 First sample row vector.
         * @param x2 Second sample row vector.
         * @param gamma RBF gamma hyperparameter.
         * @return `exp(-gamma * ||x1 - x2||^2)`.
         */
        static double radial_basis(const Matrix &x1, const Matrix &x2, double gamma = 1.0)
        {
            const double d = x1.euclideanDistance(x2);
            return std::exp(-gamma * d * d);
        }

        /**
         * @brief Dispatches to the configured kernel implementation.
         * @param x First sample row vector.
         * @param y Second sample row vector.
         * @param kernel_options Kernel selection and hyperparameters.
         * @return Kernel similarity score.
         */
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

    /**
     * @brief Kernelized SVM interface.
     *
     * Training data is expected as:
     * - `xTr`: `(num_samples, num_features)`
     * - `yTr`: `(1, num_samples)` with signed labels
     */
    class KernelSVM
    {
    private:
        KernelSVM() = default;

        Matrix xTr;                   ///< Cached training features.
        Matrix yTr;                   ///< Cached training labels.
        std::vector<Matrix> xTr_rows; ///< Row-wise training sample cache.
        Matrix weights;               ///< Learned parameter vector.
        double bias;                  ///< Learned bias term.
        double learning_rate;         ///< Optimizer learning rate.
        int max_iter;                 ///< Optimization iteration count.
        double C;                     ///< SVM hinge-loss penalty coefficient.
        KernelOptions kernel_options; ///< Kernel configuration.

    public:
        /**
         * @brief Trains a kernel SVM model.
         * @param xTr Training features.
         * @param yTr Training labels.
         * @param learning_rate Optimization step size.
         * @param max_iter Number of optimization iterations.
         * @param C Hinge-loss penalty coefficient.
         * @param kernel_options Kernel function options.
         */
        KernelSVM(Matrix xTr, Matrix yTr, double learning_rate, int max_iter, double C, KernelOptions kernel_options);

        /**
         * @brief Trains a kernel SVM with synchronous distributed gradient averaging.
         */
        static KernelSVM train_distributed(
            Matrix xTr,
            Matrix yTr,
            double learning_rate,
            int max_iter,
            double C,
            KernelOptions kernel_options,
            ctorch::distributed::ProcessGroup &group);

        /**
         * @brief Builds the symbolic training objective.
         * @return AST representing model loss.
         */
        std::shared_ptr<math::ASTNode> loss_function();

        /**
         * @brief Predicts class for one sample.
         * @param x Input sample row matrix.
         * @return Signed predicted class label.
         */
        int predict(const Matrix &x) const;

        /**
         * @brief Computes test-set classification accuracy.
         * @param xTe Test features.
         * @param yTe Test labels.
         * @return Fraction of correctly predicted samples.
         */
        double score(const Matrix &xTe, const Matrix &yTe) const;
    };
}
