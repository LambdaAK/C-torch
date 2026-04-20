#pragma once

#include <cctype>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

/**
 * @brief Token categories for a simple expression lexer.
 */
enum class TokenType {
    NUMBER,     ///< Numeric literal.
    IDENTIFIER, ///< Identifier token.
    PLUS,       ///< `+`
    MINUS,      ///< `-`
    MULTIPLY,   ///< `*`
    DIVIDE,     ///< `/`
    POWER,      ///< `^`
    EXP,        ///< `exp` function keyword.
    LOG,        ///< `log` function keyword.
    SQRT,       ///< `sqrt` function keyword.
    ABS,        ///< `abs` function keyword.
    MAX,        ///< `max` function keyword.
    MIN,        ///< `min` function keyword.
    SIGMOID,    ///< `sigmoid` function keyword.
    LPAREN,     ///< `(`
    RPAREN,     ///< `)`
    COMMA,      ///< `,`
    END         ///< End-of-input sentinel.
};

/**
 * @brief Lexical token with optional numeric or string payload.
 */
class Token {
    private:
        TokenType token_type; ///< Token category.
        std::variant<std::monostate, double, std::string> value; ///< Optional payload.

    public:

        /**
         * @brief Constructs token without payload.
         * @param token_type Token category.
         */
        Token(TokenType token_type) : token_type(token_type) {};

        /**
         * @brief Constructs numeric token.
         * @param token_type Token category.
         * @param value Numeric payload.
         */
        Token(TokenType token_type, double value) : token_type(token_type), value(value) {};

        /**
         * @brief Constructs string token.
         * @param token_type Token category.
         * @param value String payload.
         */
        Token(TokenType token_type, std::string value) : token_type(token_type), value(std::move(value)) {};

        /**
         * @brief Returns token category.
         * @return Token type enum.
         */
        TokenType getTokenType() const {
            return token_type;
        }
        
        /**
         * @brief Returns numeric payload.
         * @return Number value stored in token.
         */
        double getNumberValue() const {
            return std::get<double>(value);
        }

        /**
         * @brief Returns string payload.
         * @return String value stored in token.
         */
        std::string getStringValue() const {
            return std::get<std::string>(value);
        }

        /**
         * @brief Returns whether the token currently stores a numeric payload.
         * @return True when the token contains a `double`.
         */
        bool hasNumberValue() const {
            return std::holds_alternative<double>(value);
        }

        /**
         * @brief Returns whether the token currently stores a string payload.
         * @return True when the token contains a `std::string`.
         */
        bool hasStringValue() const {
            return std::holds_alternative<std::string>(value);
        }
};

/**
 * @brief Incremental lexer for math expressions.
 */
class Lexer {
    private:
        std::string expression; ///< Full source expression.
        size_t position;        ///< Current parsing position.

        static bool is_space(char ch) {
            return std::isspace(static_cast<unsigned char>(ch)) != 0;
        }

        static bool is_digit(char ch) {
            return std::isdigit(static_cast<unsigned char>(ch)) != 0;
        }

        static bool is_alpha(char ch) {
            return std::isalpha(static_cast<unsigned char>(ch)) != 0;
        }

        static bool is_alnum(char ch) {
            return std::isalnum(static_cast<unsigned char>(ch)) != 0;
        }

        bool atEnd() const {
            return position >= expression.size();
        }

        char currentChar() const {
            return expression[position];
        }

        void skipWhitespace() {
            while (!atEnd() && is_space(currentChar())) {
                ++position;
            }
        }

        Token readNumber() {
            const size_t start = position;
            bool seen_digit = false;
            bool seen_dot = false;
            bool seen_exp = false;

            while (!atEnd()) {
                char ch = currentChar();
                if (is_digit(ch)) {
                    seen_digit = true;
                    ++position;
                    continue;
                }
                if (ch == '.' && !seen_dot && !seen_exp) {
                    seen_dot = true;
                    ++position;
                    continue;
                }
                if ((ch == 'e' || ch == 'E') && !seen_exp && seen_digit) {
                    seen_exp = true;
                    ++position;
                    if (!atEnd() && (currentChar() == '+' || currentChar() == '-')) {
                        ++position;
                    }
                    seen_digit = false;
                    continue;
                }
                break;
            }

            if (!seen_digit) {
                throw std::invalid_argument("Lexer::getNextToken(): malformed numeric literal.");
            }

            const std::string text = expression.substr(start, position - start);
            size_t consumed = 0;
            const double value = std::stod(text, &consumed);
            if (consumed != text.size()) {
                throw std::invalid_argument("Lexer::getNextToken(): malformed numeric literal.");
            }
            return Token(TokenType::NUMBER, value);
        }

        Token readIdentifier() {
            const size_t start = position;
            while (!atEnd() && (is_alnum(currentChar()) || currentChar() == '_')) {
                ++position;
            }

            const std::string name = expression.substr(start, position - start);
            if (name == "exp") {
                return Token(TokenType::EXP);
            }
            if (name == "log") {
                return Token(TokenType::LOG);
            }
            if (name == "sqrt") {
                return Token(TokenType::SQRT);
            }
            if (name == "abs") {
                return Token(TokenType::ABS);
            }
            if (name == "max") {
                return Token(TokenType::MAX);
            }
            if (name == "min") {
                return Token(TokenType::MIN);
            }
            if (name == "sigmoid") {
                return Token(TokenType::SIGMOID);
            }

            return Token(TokenType::IDENTIFIER, name);
        }

    public:
        /**
         * @brief Creates lexer for a source expression.
         * @param expression Input expression string.
         */
        Lexer(std::string expression) : expression(std::move(expression)), position(0) {}

        /**
         * @brief Returns next token from the stream.
         * @return Next token, or `TokenType::END` when input is exhausted.
         */
        Token getNextToken() {
            skipWhitespace();

            if (atEnd()) {
                return Token(TokenType::END);
            }

            const char ch = currentChar();
            if (is_digit(ch) || (ch == '.' && position + 1 < expression.size() && is_digit(expression[position + 1]))) {
                return readNumber();
            }
            if (is_alpha(ch) || ch == '_') {
                return readIdentifier();
            }

            ++position;
            switch (ch) {
                case '+':
                    return Token(TokenType::PLUS);
                case '-':
                    return Token(TokenType::MINUS);
                case '*':
                    return Token(TokenType::MULTIPLY);
                case '/':
                    return Token(TokenType::DIVIDE);
                case '^':
                    return Token(TokenType::POWER);
                case '(':
                    return Token(TokenType::LPAREN);
                case ')':
                    return Token(TokenType::RPAREN);
                case ',':
                    return Token(TokenType::COMMA);
                default:
                    throw std::invalid_argument(
                        "Lexer::getNextToken(): unexpected character '" + std::string(1, ch) +
                        "' at position " + std::to_string(position - 1));
            }
        }

};
