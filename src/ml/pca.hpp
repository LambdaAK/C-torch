#pragma once

#include <cstddef>
#include <random>

#include "../math/matrix.hpp"

namespace ml {

  class PCA {

    private:
      Matrix Cov;
      std::pair<Matrix, Matrix> householder_reflection(const Matrix& X);
      float l2_norm(const Matrix& vec);
      Matrix l2_normalize(const Matrix& vec);

    public:
      // pass in centered X
      PCA(const Matrix& X);
      Matrix compute_projection_mat(int k, int max_iter = 1000, float tol = 1e-9);
  };

}