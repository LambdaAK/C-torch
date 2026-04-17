/*

    AST

    NumberNode
        - Number
    VariableNode
        - Name
    BinaryOpNode
        - Operator
        - Left
        - Right
    UnaryOpNode
        - Operator
        - Operand

*/

#pragma once
#include <string>
#include <cmath>
#include <set>


namespace math {

    class ASTNode;
    class NumberNode;
    class VariableNode;
    class AppNode;

    inline std::shared_ptr<ASTNode> Num(double value);
    inline std::shared_ptr<ASTNode> Var(std::string name);

    inline std::shared_ptr<ASTNode> operator+(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right);
    inline std::shared_ptr<ASTNode> operator-(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right);
    inline std::shared_ptr<ASTNode> operator*(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right);
    inline std::shared_ptr<ASTNode> operator/(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right);
    inline std::shared_ptr<ASTNode> operator^(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right);

    inline std::shared_ptr<ASTNode> Exp(std::shared_ptr<ASTNode> operand);
    inline std::shared_ptr<ASTNode> Log(std::shared_ptr<ASTNode> operand);
    inline std::shared_ptr<ASTNode> Sqrt(std::shared_ptr<ASTNode> operand);
    inline std::shared_ptr<ASTNode> Abs(std::shared_ptr<ASTNode> operand);

    class ASTNode {
        public:
            virtual ~ASTNode() = default;

            virtual std::string to_string() const {
                return "ASTNode";
            }
            virtual std::shared_ptr<ASTNode> simplify() const {
                throw std::runtime_error("Not implemented");
            }
            virtual std::shared_ptr<ASTNode> substitute (std::string var_name, std::shared_ptr<ASTNode> replace_with) {
                throw std::runtime_error("Not implemented");
            }
            virtual std::set<std::string> variables() const {
                throw std::runtime_error("Not implemented");
            }
    };

    class NumberNode : public ASTNode {
        private:
            double value;
        public:
            NumberNode(double value) : value(value) {}
            double getValue() const {
                return value;
            }
            std::string to_string() const {
                return std::to_string(value);
            }
            std::shared_ptr<ASTNode> simplify() const {
                // a number is already simplified
                return std::make_shared<NumberNode>(value);                
            }
            
            std::shared_ptr<ASTNode> substitute(std::string var_name, std::shared_ptr<ASTNode> replace_with) {
                return std::make_shared<NumberNode>(value);
            }

            std::set<std::string> variables() const {
                // a number node has no variables
                return {};
            }
    };

    class VariableNode : public ASTNode {
        private:
            std::string name;

        public:
            VariableNode(std::string name) : name(name) {}
            std::string getName() const {
                return name;
            }
            std::string to_string() const {
                return name;
            }
            std::shared_ptr<ASTNode> simplify() const {
                return std::make_shared<VariableNode>(name);
            }
            std::shared_ptr<ASTNode> substitute(std::string var_name, std::shared_ptr<ASTNode> replace_with) {
                // if the var name equals, then return replace_with
                if (var_name == name) {
                    return replace_with;
                }
                // otherwise, return a copy of this variable node
                return std::make_shared<VariableNode>(name);
            }

            std::set<std::string> variables() const {
                // just itself
                return {name};
            }
    };

    class AppNode : public ASTNode {
        private:
            std::string function;
            std::vector<std::shared_ptr<ASTNode>> args;
        public:
            AppNode(std::string function, std::vector<std::shared_ptr<ASTNode>> args) : function(function), args(args) {
                // check that the function is valid
                /*
                    Functions:
                        - exp
                        - log
                        - max
                        - min
                        - sqrt
                        - abs

                        Binary operations:

                        Plus
                        Minus
                        Multiply
                        Divide
                        Power

                */

            }

            std::string getFunction() const {
                return function;
            }

            std::vector<std::shared_ptr<ASTNode>> getArgs() const {
                return args;
            }

            std::string to_string() const {
                
                // if it's +, -, *, /, or ^, print it as an infix operator with parenthesis

                if (function == "+" || function == "-" || function == "*" || function == "/" || function == "^") {
                    return "(" + args[0]->to_string() + " " + function + " " + args[1]->to_string() + ")";
                }

                // otherwise, print it as a function application

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

            std::shared_ptr<ASTNode> simplify() const {
                // simplify each argument
                std::vector<std::shared_ptr<ASTNode>> simplified_args;
                for (const auto &arg : args) {
                    simplified_args.push_back(arg->simplify());
                }

                // if all arguments are numbers, we can simplify further
                bool all_args_numbers = true;
                for (const auto &arg : simplified_args) {
                    if (!std::dynamic_pointer_cast<NumberNode>(arg)) {
                        all_args_numbers = false;
                        break;
                    }
                }

                if (all_args_numbers) {
                    // get the values of the arguments
                    std::vector<double> arg_values;
                    for (const auto &arg : simplified_args) {
                        auto cast = std::dynamic_pointer_cast<NumberNode>(arg);
                        arg_values.push_back(cast->getValue());
                    }

                    // apply the function

                    if (function == "exp") return Num(std::exp(arg_values[0]));
                    if (function == "log") return Num(std::log(arg_values[0]));
                    if (function == "max") {
                        // take the maximum of all arguments
                        double max_value = arg_values[0];
                        for (size_t i = 1; i < arg_values.size(); i++) {
                            max_value = std::max(max_value, arg_values[i]);
                        }
                        return Num(max_value);
                    }
                    if (function == "min") {
                        // take the minimum of all arguments
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

                    else throw std::runtime_error("Unknown function: " + function);
                }
                
                // if we reach this point, we cannot simplify any further

                return std::make_shared<AppNode>(function, simplified_args);
            }

            std::shared_ptr<ASTNode> substitute(std::string var_name, std::shared_ptr<ASTNode> replace_with) {
                // substitute in each argument
                std::vector<std::shared_ptr<ASTNode>> new_args;
                for (const auto &arg : args) {
                    new_args.push_back(arg->substitute(var_name, replace_with));
                }
                return std::make_shared<AppNode>(function, new_args);
            }

            std::set<std::string> variables() const {
                std::set<std::string> vars;
                for (const auto &arg : args) {
                    auto arg_vars = arg->variables();
                    vars.insert(arg_vars.begin(), arg_vars.end());
                }
                return vars;
            }

            
    };

    /*
        Functions and infix operators for constructing the AST
    */

    // Leaf nodes

    inline std::shared_ptr<ASTNode> Num(double value) {
        return std::make_shared<NumberNode>(value);
    }

    inline std::shared_ptr<ASTNode> Var(std::string name) {
        return std::make_shared<VariableNode>(name);
    }

    // Infix operators
    
    inline std::shared_ptr<ASTNode> operator+(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right) {
        return std::make_shared<AppNode>("+", std::vector<std::shared_ptr<ASTNode>>{left, right});
    }

    inline std::shared_ptr<ASTNode> operator-(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right) {
        return std::make_shared<AppNode>("-", std::vector<std::shared_ptr<ASTNode>>{left, right});
    }

    inline std::shared_ptr<ASTNode> operator*(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right) {
        return std::make_shared<AppNode>("*", std::vector<std::shared_ptr<ASTNode>>{left, right});
    }

    inline std::shared_ptr<ASTNode> operator/(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right) {
        return std::make_shared<AppNode>("/", std::vector<std::shared_ptr<ASTNode>>{left, right});
    }

    inline std::shared_ptr<ASTNode> operator^(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right) {
        return std::make_shared<AppNode>("^", std::vector<std::shared_ptr<ASTNode>>{left, right});
    }

    inline std::shared_ptr<ASTNode> operator-(std::shared_ptr<ASTNode> node) {
        return Num(-1) * node;
    }

    // Functions

    inline std::shared_ptr<ASTNode> Exp(std::shared_ptr<ASTNode> operand) {
        std::vector<std::shared_ptr<ASTNode>> args;
        args.push_back(operand);
        return std::make_shared<AppNode>("exp", args);
    }

    inline std::shared_ptr<ASTNode> Log(std::shared_ptr<ASTNode> operand) {
        std::vector<std::shared_ptr<ASTNode>> args;
        args.push_back(operand);
        return std::make_shared<AppNode>("log", args);
    }

    inline std::shared_ptr<ASTNode> Sqrt(std::shared_ptr<ASTNode> operand) {
        std::vector<std::shared_ptr<ASTNode>> args;
        args.push_back(operand);
        return std::make_shared<AppNode>("sqrt", args);
    }

    inline std::shared_ptr<ASTNode> Abs(std::shared_ptr<ASTNode> operand) {
        std::vector<std::shared_ptr<ASTNode>> args;
        args.push_back(operand);
        return std::make_shared<AppNode>("abs", args);
    }

    inline std::shared_ptr<ASTNode> Max(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right) {
        std::vector<std::shared_ptr<ASTNode>> args;
        args.push_back(left);
        args.push_back(right);
        return std::make_shared<AppNode>("max", args);
    }

    inline std::shared_ptr<ASTNode> Min(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right) {
        std::vector<std::shared_ptr<ASTNode>> args;
        args.push_back(left);
        args.push_back(right);
        return std::make_shared<AppNode>("min", args);
    }

    inline std::shared_ptr<ASTNode> Sigmoid(std::shared_ptr<ASTNode> operand) {
        return Num(1) / (Num(1) + Exp(-operand));
    }

}

