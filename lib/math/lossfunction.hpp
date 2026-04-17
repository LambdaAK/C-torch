#pragma once
#include "matrix.hpp"
#include "ast.hpp"
#include <string>


    /**
     * @brief Interface for supervised learning losses used by optimizers.
     */
    class LossFunction {
        public:
            /**
             * @brief Virtual destructor for interface-safe deletion.
             */
            virtual ~LossFunction() = default;

            /**
             * @brief Returns per-sample symbolic loss expression.
             * @param x Single input sample row matrix.
             * @param y Corresponding label.
             * @return AST encoding sample loss.
             */
            virtual std::shared_ptr<math::ASTNode> sample_loss(const Matrix &x, int y) const = 0;

            /**
             * @brief Returns symbolic regularization term.
             * @return AST regularizer expression (or zero).
             */
            virtual std::shared_ptr<math::ASTNode> regularizer() const = 0;
    };
