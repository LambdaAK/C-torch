/**
 * @file ast.hpp
 * @brief Symbolic expression tree primitives and helper constructors.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace math {

    class ASTNode;
    class NumberNode;
    class VariableNode;
    class AppNode;

    /**
     * @brief Constructs a numeric constant node.
     * @param value Constant value.
     * @return Shared pointer to `NumberNode`.
     */
    inline std::shared_ptr<ASTNode> Num(double value);

    /**
     * @brief Constructs a variable node.
     * @param name Variable identifier.
     * @return Shared pointer to `VariableNode`.
     */
    inline std::shared_ptr<ASTNode> Var(std::string name);

    /**
     * @brief Constructs addition node.
     * @param left Left operand.
     * @param right Right operand.
     * @return Shared pointer to application node.
     */
    inline std::shared_ptr<ASTNode> operator+(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right);

    /**
     * @brief Constructs subtraction node.
     * @param left Left operand.
     * @param right Right operand.
     * @return Shared pointer to application node.
     */
    inline std::shared_ptr<ASTNode> operator-(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right);

    /**
     * @brief Constructs multiplication node.
     * @param left Left operand.
     * @param right Right operand.
     * @return Shared pointer to application node.
     */
    inline std::shared_ptr<ASTNode> operator*(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right);

    /**
     * @brief Constructs division node.
     * @param left Left operand.
     * @param right Right operand.
     * @return Shared pointer to application node.
     */
    inline std::shared_ptr<ASTNode> operator/(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right);

    /**
     * @brief Constructs exponentiation node.
     * @param left Base operand.
     * @param right Exponent operand.
     * @return Shared pointer to application node.
     */
    inline std::shared_ptr<ASTNode> operator^(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right);

    /**
     * @brief Constructs `exp` function application.
     * @param operand Function argument.
     * @return Shared pointer to application node.
     */
    inline std::shared_ptr<ASTNode> Exp(std::shared_ptr<ASTNode> operand);

    /**
     * @brief Constructs `log` function application.
     * @param operand Function argument.
     * @return Shared pointer to application node.
     */
    inline std::shared_ptr<ASTNode> Log(std::shared_ptr<ASTNode> operand);

    /**
     * @brief Constructs `sqrt` function application.
     * @param operand Function argument.
     * @return Shared pointer to application node.
     */
    inline std::shared_ptr<ASTNode> Sqrt(std::shared_ptr<ASTNode> operand);

    /**
     * @brief Constructs `abs` function application.
     * @param operand Function argument.
     * @return Shared pointer to application node.
     */
    inline std::shared_ptr<ASTNode> Abs(std::shared_ptr<ASTNode> operand);

    /**
     * @brief Base class for symbolic AST nodes.
     */
    class ASTNode {
        public:
            /**
             * @brief Virtual destructor for polymorphic usage.
             */
            virtual ~ASTNode() = default;

            /**
             * @brief Serializes node to readable expression string.
             * @return String representation of the AST node.
             */
            virtual std::string to_string() const {
                return "ASTNode";
            }

            /**
             * @brief Simplifies this node recursively.
             * @return Simplified AST node.
             */
            virtual std::shared_ptr<ASTNode> simplify() const {
                throw std::runtime_error("Not implemented");
            }

            /**
             * @brief Substitutes one variable with another expression.
             * @param var_name Variable name to replace.
             * @param replace_with Replacement expression.
             * @return New AST node with substitution applied.
             */
            virtual std::shared_ptr<ASTNode> substitute (std::string var_name, std::shared_ptr<ASTNode> replace_with) {
                throw std::runtime_error("Not implemented");
            }

            /**
             * @brief Returns variable symbols referenced by this expression.
             * @return Set of variable names.
             */
            virtual std::set<std::string> variables() const {
                throw std::runtime_error("Not implemented");
            }
    };

    /**
     * @brief Numeric leaf node.
     */
    class NumberNode : public ASTNode {
        private:
            double value; ///< Stored numeric constant.
        public:
            /**
             * @brief Constructs a numeric node.
             * @param value Constant value.
             */
            NumberNode(double value) : value(value) {}

            /**
             * @brief Returns numeric value.
             * @return Stored constant.
             */
            double getValue() const {
                return value;
            }

            /**
             * @brief Converts node to string.
             * @return Numeric value as string.
             */
            std::string to_string() const {
                return std::to_string(value);
            }

            /**
             * @brief Number nodes are already simplified.
             * @return Copy of this number node.
             */
            std::shared_ptr<ASTNode> simplify() const {
                return std::make_shared<NumberNode>(value);                
            }
            
            /**
             * @brief Substitution does not affect constants.
             * @param var_name Unused variable name.
             * @param replace_with Unused replacement node.
             * @return Copy of this number node.
             */
            std::shared_ptr<ASTNode> substitute(std::string var_name, std::shared_ptr<ASTNode> replace_with) {
                (void)var_name;
                (void)replace_with;
                return std::make_shared<NumberNode>(value);
            }

            /**
             * @brief Constants contain no variables.
             * @return Empty set.
             */
            std::set<std::string> variables() const {
                return {};
            }
    };

    /**
     * @brief Variable leaf node.
     */
    class VariableNode : public ASTNode {
        private:
            std::string name; ///< Variable identifier.

        public:
            /**
             * @brief Constructs a variable node.
             * @param name Variable identifier.
             */
            VariableNode(std::string name) : name(name) {}

            /**
             * @brief Returns variable identifier.
             * @return Variable name.
             */
            std::string getName() const {
                return name;
            }

            /**
             * @brief Converts node to string.
             * @return Variable identifier.
             */
            std::string to_string() const {
                return name;
            }

            /**
             * @brief Variable is already in simplest symbolic form.
             * @return Copy of variable node.
             */
            std::shared_ptr<ASTNode> simplify() const {
                return std::make_shared<VariableNode>(name);
            }

            /**
             * @brief Replaces variable when names match.
             * @param var_name Target variable to replace.
             * @param replace_with Replacement expression.
             * @return Replacement expression when matched, else copy of this node.
             */
            std::shared_ptr<ASTNode> substitute(std::string var_name, std::shared_ptr<ASTNode> replace_with) {
                if (var_name == name) {
                    return replace_with;
                }
                return std::make_shared<VariableNode>(name);
            }

            /**
             * @brief Returns this variable as singleton set.
             * @return Set containing `name`.
             */
            std::set<std::string> variables() const {
                return {name};
            }
    };

    /**
     * @brief Generic function or operator application node.
     */
    class AppNode : public ASTNode {
        private:
            std::string function; ///< Function/operator name.
            std::vector<std::shared_ptr<ASTNode>> args; ///< Ordered operands.
        public:
            /**
             * @brief Creates a function application node.
             * @param function Function or operator name.
             * @param args Ordered operands.
             */
            AppNode(std::string function, std::vector<std::shared_ptr<ASTNode>> args) : function(function), args(args) {}

            /**
             * @brief Returns function/operator name.
             * @return Function identifier.
             */
            std::string getFunction() const {
                return function;
            }

            /**
             * @brief Returns argument list.
             * @return Vector of argument AST pointers.
             */
            std::vector<std::shared_ptr<ASTNode>> getArgs() const {
                return args;
            }

            /**
             * @brief Converts node to string.
             * @return Readable infix/function expression.
             */
            std::string to_string() const {
                if (function == "+" || function == "-" || function == "*" || function == "/" || function == "^") {
                    return "(" + args[0]->to_string() + " " + function + " " + args[1]->to_string() + ")";
                }

                std::string result = function + "(";
                for (size_t i = 0; i < args.size(); i++) {
                    result += args[i]->to_string();
                    if (i < args.size() - 1) {
                        result += ", ";
                    }
                }
                result += ")";
                return result;
            }

            /**
             * @brief Simplifies node by simplifying arguments and folding constants.
             * @return Simplified expression node.
             */
            std::shared_ptr<ASTNode> simplify() const {
                std::vector<std::shared_ptr<ASTNode>> simplified_args;
                for (const auto &arg : args) {
                    simplified_args.push_back(arg->simplify());
                }

                bool all_args_numbers = true;
                for (const auto &arg : simplified_args) {
                    if (!std::dynamic_pointer_cast<NumberNode>(arg)) {
                        all_args_numbers = false;
                        break;
                    }
                }

                if (all_args_numbers) {
                    std::vector<double> arg_values;
                    for (const auto &arg : simplified_args) {
                        auto cast = std::dynamic_pointer_cast<NumberNode>(arg);
                        arg_values.push_back(cast->getValue());
                    }

                    if (function == "exp") return Num(std::exp(arg_values[0]));
                    if (function == "log") return Num(std::log(arg_values[0]));
                    if (function == "max") {
                        double max_value = arg_values[0];
                        for (size_t i = 1; i < arg_values.size(); i++) {
                            max_value = std::max(max_value, arg_values[i]);
                        }
                        return Num(max_value);
                    }
                    if (function == "min") {
                        double min_value = arg_values[0];
                        for (size_t i = 1; i < arg_values.size(); i++) {
                            min_value = std::min(min_value, arg_values[i]);
                        }
                        return Num(min_value);
                    }
                    if (function == "sqrt") return Num(std::sqrt(arg_values[0]));
                    if (function == "abs") return Num(std::abs(arg_values[0]));
                    if (function == "+") return Num(arg_values[0] + arg_values[1]);
                    if (function == "-") return Num(arg_values[0] - arg_values[1]);
                    if (function == "*") return Num(arg_values[0] * arg_values[1]);
                    if (function == "/") return Num(arg_values[0] / arg_values[1]);
                    if (function == "^") return Num(std::pow(arg_values[0], arg_values[1]));

                    throw std::runtime_error("Unknown function: " + function);
                }
                
                return std::make_shared<AppNode>(function, simplified_args);
            }

            /**
             * @brief Applies variable substitution recursively to all arguments.
             * @param var_name Variable name to replace.
             * @param replace_with Replacement expression.
             * @return New application node with substitutions applied.
             */
            std::shared_ptr<ASTNode> substitute(std::string var_name, std::shared_ptr<ASTNode> replace_with) {
                std::vector<std::shared_ptr<ASTNode>> new_args;
                for (const auto &arg : args) {
                    new_args.push_back(arg->substitute(var_name, replace_with));
                }
                return std::make_shared<AppNode>(function, new_args);
            }

            /**
             * @brief Collects variables referenced by all arguments.
             * @return Union of variable sets from arguments.
             */
            std::set<std::string> variables() const {
                std::set<std::string> vars;
                for (const auto &arg : args) {
                    auto arg_vars = arg->variables();
                    vars.insert(arg_vars.begin(), arg_vars.end());
                }
                return vars;
            }
    };

    /**
     * @brief Creates numeric constant node.
     * @param value Constant value.
     * @return Number node.
     */
    inline std::shared_ptr<ASTNode> Num(double value) {
        return std::make_shared<NumberNode>(value);
    }

    /**
     * @brief Creates variable node.
     * @param name Variable identifier.
     * @return Variable node.
     */
    inline std::shared_ptr<ASTNode> Var(std::string name) {
        return std::make_shared<VariableNode>(name);
    }

    /**
     * @brief Creates addition expression.
     * @param left Left operand.
     * @param right Right operand.
     * @return Application node representing addition.
     */
    inline std::shared_ptr<ASTNode> operator+(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right) {
        return std::make_shared<AppNode>("+", std::vector<std::shared_ptr<ASTNode>>{left, right});
    }

    /**
     * @brief Creates subtraction expression.
     * @param left Left operand.
     * @param right Right operand.
     * @return Application node representing subtraction.
     */
    inline std::shared_ptr<ASTNode> operator-(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right) {
        return std::make_shared<AppNode>("-", std::vector<std::shared_ptr<ASTNode>>{left, right});
    }

    /**
     * @brief Creates multiplication expression.
     * @param left Left operand.
     * @param right Right operand.
     * @return Application node representing multiplication.
     */
    inline std::shared_ptr<ASTNode> operator*(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right) {
        return std::make_shared<AppNode>("*", std::vector<std::shared_ptr<ASTNode>>{left, right});
    }

    /**
     * @brief Creates division expression.
     * @param left Left operand.
     * @param right Right operand.
     * @return Application node representing division.
     */
    inline std::shared_ptr<ASTNode> operator/(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right) {
        return std::make_shared<AppNode>("/", std::vector<std::shared_ptr<ASTNode>>{left, right});
    }

    /**
     * @brief Creates power expression.
     * @param left Base operand.
     * @param right Exponent operand.
     * @return Application node representing exponentiation.
     */
    inline std::shared_ptr<ASTNode> operator^(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right) {
        return std::make_shared<AppNode>("^", std::vector<std::shared_ptr<ASTNode>>{left, right});
    }

    /**
     * @brief Creates unary negation expression.
     * @param node Operand expression.
     * @return Expression equivalent to `-node`.
     */
    inline std::shared_ptr<ASTNode> operator-(std::shared_ptr<ASTNode> node) {
        return Num(-1) * node;
    }

    /**
     * @brief Creates exponential function expression.
     * @param operand Function argument.
     * @return Application node for `exp`.
     */
    inline std::shared_ptr<ASTNode> Exp(std::shared_ptr<ASTNode> operand) {
        std::vector<std::shared_ptr<ASTNode>> args;
        args.push_back(operand);
        return std::make_shared<AppNode>("exp", args);
    }

    /**
     * @brief Creates natural logarithm function expression.
     * @param operand Function argument.
     * @return Application node for `log`.
     */
    inline std::shared_ptr<ASTNode> Log(std::shared_ptr<ASTNode> operand) {
        std::vector<std::shared_ptr<ASTNode>> args;
        args.push_back(operand);
        return std::make_shared<AppNode>("log", args);
    }

    /**
     * @brief Creates square-root function expression.
     * @param operand Function argument.
     * @return Application node for `sqrt`.
     */
    inline std::shared_ptr<ASTNode> Sqrt(std::shared_ptr<ASTNode> operand) {
        std::vector<std::shared_ptr<ASTNode>> args;
        args.push_back(operand);
        return std::make_shared<AppNode>("sqrt", args);
    }

    /**
     * @brief Creates absolute-value function expression.
     * @param operand Function argument.
     * @return Application node for `abs`.
     */
    inline std::shared_ptr<ASTNode> Abs(std::shared_ptr<ASTNode> operand) {
        std::vector<std::shared_ptr<ASTNode>> args;
        args.push_back(operand);
        return std::make_shared<AppNode>("abs", args);
    }

    /**
     * @brief Creates max function expression.
     * @param left First argument.
     * @param right Second argument.
     * @return Application node for `max(left, right)`.
     */
    inline std::shared_ptr<ASTNode> Max(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right) {
        std::vector<std::shared_ptr<ASTNode>> args;
        args.push_back(left);
        args.push_back(right);
        return std::make_shared<AppNode>("max", args);
    }

    /**
     * @brief Creates min function expression.
     * @param left First argument.
     * @param right Second argument.
     * @return Application node for `min(left, right)`.
     */
    inline std::shared_ptr<ASTNode> Min(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right) {
        std::vector<std::shared_ptr<ASTNode>> args;
        args.push_back(left);
        args.push_back(right);
        return std::make_shared<AppNode>("min", args);
    }

    /**
     * @brief Creates sigmoid expression using primitive AST operators.
     * @param operand Input expression.
     * @return Expression `1 / (1 + exp(-operand))`.
     */
    inline std::shared_ptr<ASTNode> Sigmoid(std::shared_ptr<ASTNode> operand) {
        return Num(1) / (Num(1) + Exp(-operand));
    }

} // namespace math

