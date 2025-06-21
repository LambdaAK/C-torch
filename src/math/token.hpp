#include <variant>
#include <string>

enum class TokenType {
    // Literals

    NUMBER,
    IDENTIFIER,

    // Operators

    PLUS,
    MINUS,
    MULTIPLY,
    DIVIDE,
    POWER,

    // Functions
    EXP,
    LOG,
    SQRT,
    ABS,

    // Other stuff

    LPAREN,
    RPAREN,
    COMMA,
    END

};

class Token {
    private:
        TokenType token_type;
        std::variant<std::monostate, double, std::string> value;

    public:

        Token(TokenType token_type) : token_type(token_type) {};

        Token(TokenType token_type, double value) : token_type(token_type), value(value) {};

        Token(TokenType token_type, std::string value) : token_type(token_type), value(value) {};

        TokenType getTokenType() const {
            return token_type;
        }
        
        double getNumberValue() const {
            return std::get<double>(value);
        }
        std::string getStringValue() const {
            return std::get<std::string>(value);
        }
};

class Lexer {
    private:
        std::string expression;
        size_t position;

    public:
        Lexer(std::string expression) : expression(expression), position(0) {}
        Token getNextToken() {
            // if there isn't any more input, return EOF
            if (position >= expression.size()) {
                return Token(TokenType::END);
            }

        }

};