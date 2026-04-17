#pragma once
#include "matrix.hpp"
#include "ast.hpp"
#include <string>


    class LossFunction {
        public:
            virtual ~LossFunction() = default;
            virtual std::shared_ptr<math::ASTNode> sample_loss(const Matrix &x, int y) const = 0;
            virtual std::shared_ptr<math::ASTNode> regularizer() const = 0;
    };
