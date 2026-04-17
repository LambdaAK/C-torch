#pragma once

#include <cstddef>
#include <random>

#include "math/matrix.hpp"

namespace ml {

  /**
   * @brief Principal Component Analysis helper built from covariance matrix.
   *
   * The constructor expects features that are already column-centered.
   */
  class PCA {

    private:
      Matrix Cov; ///< Cached covariance matrix of centered input data.

      /**
       * @brief Computes Householder QR factorization step.
       * @param X Input matrix.
       * @return Pair `(Q, R)` from one Householder decomposition pass.
       */
      std::pair<Matrix, Matrix> householder_reflection(const Matrix& X);

      /**
       * @brief Computes Euclidean norm of a vector represented as a matrix.
       * @param vec Input vector matrix.
       * @return L2 norm value.
       */
      float l2_norm(const Matrix& vec);

      /**
       * @brief Returns unit-norm version of a vector matrix.
       * @param vec Input vector matrix.
       * @return L2-normalized vector matrix.
       */
      Matrix l2_normalize(const Matrix& vec);

    public:
      /**
       * @brief Builds covariance representation from centered data.
       * @param X Centered feature matrix `(num_samples, num_features)`.
       */
      PCA(const Matrix& X);

      /**
       * @brief Computes top-`k` projection matrix using iterative eigensolver logic.
       * @param k Number of principal components.
       * @param max_iter Maximum iterations for convergence.
       * @param tol Convergence tolerance.
       * @return Projection matrix with `k` principal directions.
       */
      Matrix compute_projection_mat(int k, int max_iter = 1000, float tol = 1e-9);
  };

}
