#include <variant>
#include <string>

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
        Token(TokenType token_type, std::string value) : token_type(token_type), value(value) {};

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
};

/**
 * @brief Incremental lexer for math expressions.
 */
class Lexer {
    private:
        std::string expression; ///< Full source expression.
        size_t position;        ///< Current parsing position.

    public:
        /**
         * @brief Creates lexer for a source expression.
         * @param expression Input expression string.
         */
        Lexer(std::string expression) : expression(expression), position(0) {}

        /**
         * @brief Returns next token from the stream.
         * @return Next token, or `TokenType::END` when input is exhausted.
         */
        Token getNextToken() {
            // if there isn't any more input, return EOF
            if (position >= expression.size()) {
                return Token(TokenType::END);
            }

        }

};
