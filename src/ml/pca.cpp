#include "pca.hpp"

namespace ml {

  PCA::PCA(const Matrix& X) {
    // Add small regularization for numerical stability
    float reg = (X.numRows() < 200) ? 1e-5 : 0;
    Cov = 1.0/(X.numRows() - 1) * X.transpose() * X + reg * Matrix::eye(X.numCols());
  }

  float PCA::l2_norm(const Matrix& vec) {
    float norm = 0.0f;
    for (size_t i = 0; i < vec.numRows(); ++i) {
      norm += vec(i, 0) * vec(i, 0);
    }

    return std::sqrt(norm);
  }

  Matrix PCA::l2_normalize(const Matrix& vec) {
    float norm = l2_norm(vec);

    if (norm < 1e-10) {
      std::cerr << "Warning: Very small norm detected!" << std::endl;
      norm += 1e-10;
    }

    return (1/norm) * vec;
  }

  std::pair<Matrix, Matrix> PCA::householder_reflection(const Matrix& X) {
    Matrix R = X;
    Matrix Q = Matrix::eye(X.numRows());
    for (size_t k = 0; k < std::min(X.numRows() - 1, X.numCols()); ++k) {
      Matrix x(X.numRows() - k, 1);
      size_t j = k;
      for (size_t i = 0; i < x.numRows(); ++i) {
        x(i, 0) = R(j, k);
        j += 1;
      }

      Matrix e(x.numRows(), 1);
      e(0, 0) = 1;
      
      Matrix v;
      if (x(0, 0) >= 0) {
        v = x + (l2_norm(x) * e);
      }
      else {
        v = x - (l2_norm(x) * e);
      }

      v = l2_normalize(v);

      Matrix Hk = Matrix::eye(v.numRows()) - (2 * (v * v.transpose()));

      Matrix H = Matrix::eye(X.numRows());

      for (size_t i = 0; i < Hk.numRows(); ++i) {
        for (size_t j = 0; j < Hk.numRows(); ++j) {
          H(i + k, j + k) = Hk(i, j);
        }
      }

      R = H * R;
      Q = Q * H.transpose();
    }

    for (size_t i = 0; i < R.numRows(); ++i) {
      for (size_t j = 0; j < R.numCols(); ++j) {
        if (i > j) {
          R(i, j) = 0;
        }
      }
    }

    return {Q, R};
  }

  Matrix PCA::compute_projection_mat(int k, int max_iter, float tol) {
    size_t n = Cov.numRows();
    Matrix Ak = Cov;
    Matrix Q_total = Matrix::eye(n);

    for (int i = 0; i < max_iter; ++i) {
      auto [Q, R] = householder_reflection(Ak);
      Ak = R * Q;
      Q_total = Q_total * Q;

      float off_diag_sum = 0.0;
      for (size_t r = 0; r < n; ++r)
        for (size_t c = 0; c < n; ++c)
          if (r != c)
            off_diag_sum += std::abs(Ak(r, c));
      if (off_diag_sum < tol) {
        std::cout << "Hit target tolerance." << std::endl;
        break;
      }
    }

    std::vector<float> eigenvalues;
    for (size_t i = 0; i < n; ++i) {
      eigenvalues.push_back(Ak(i, i));
    }

    std::vector<size_t> idx(eigenvalues.size());
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
      idx[i] = i;
    }

    std::sort(idx.begin(), idx.end(),
        [&eigenvalues](size_t i1, size_t i2) { return std::abs(eigenvalues[i1]) > std::abs(eigenvalues[i2]); });

    Matrix eigenvectors = Q_total;
    
    Matrix proj_mat(eigenvectors.numRows(), k);
    for (size_t i = 0; i < eigenvectors.numRows(); ++i) {
      for (size_t j = 0; j < k; ++j) {
        proj_mat(i, j) = eigenvectors(i, idx[j]);
      }
    }

    return proj_mat;
  }

}