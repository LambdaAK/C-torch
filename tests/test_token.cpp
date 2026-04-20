#include <gtest/gtest.h>

#include <string>

#include "math/token.hpp"

namespace {

void expect_token(Lexer& lexer, TokenType expected_type) {
    const Token token = lexer.getNextToken();
    EXPECT_EQ(token.getTokenType(), expected_type);
}

void expect_identifier(Lexer& lexer, const std::string& expected_name) {
    const Token token = lexer.getNextToken();
    EXPECT_EQ(token.getTokenType(), TokenType::IDENTIFIER);
    EXPECT_TRUE(token.hasStringValue());
    EXPECT_EQ(token.getStringValue(), expected_name);
}

void expect_number(Lexer& lexer, double expected_value) {
    const Token token = lexer.getNextToken();
    EXPECT_EQ(token.getTokenType(), TokenType::NUMBER);
    EXPECT_TRUE(token.hasNumberValue());
    EXPECT_DOUBLE_EQ(token.getNumberValue(), expected_value);
}

} // namespace

TEST(TokenLexer, TokenizesKeywordsIdentifiersAndNumbers) {
    Lexer lexer("exp(x) + max(1.25, _y2) - sigmoid(z) / min(abs(.5), log2)");

    expect_token(lexer, TokenType::EXP);
    expect_token(lexer, TokenType::LPAREN);
    expect_identifier(lexer, "x");
    expect_token(lexer, TokenType::RPAREN);
    expect_token(lexer, TokenType::PLUS);
    expect_token(lexer, TokenType::MAX);
    expect_token(lexer, TokenType::LPAREN);
    expect_number(lexer, 1.25);
    expect_token(lexer, TokenType::COMMA);
    expect_identifier(lexer, "_y2");
    expect_token(lexer, TokenType::RPAREN);
    expect_token(lexer, TokenType::MINUS);
    expect_token(lexer, TokenType::SIGMOID);
    expect_token(lexer, TokenType::LPAREN);
    expect_identifier(lexer, "z");
    expect_token(lexer, TokenType::RPAREN);
    expect_token(lexer, TokenType::DIVIDE);
    expect_token(lexer, TokenType::MIN);
    expect_token(lexer, TokenType::LPAREN);
    expect_token(lexer, TokenType::ABS);
    expect_token(lexer, TokenType::LPAREN);
    expect_number(lexer, 0.5);
    expect_token(lexer, TokenType::RPAREN);
    expect_token(lexer, TokenType::COMMA);
    expect_identifier(lexer, "log2");
    expect_token(lexer, TokenType::RPAREN);
    expect_token(lexer, TokenType::END);
}

TEST(TokenLexer, RejectsUnexpectedCharacters) {
    Lexer lexer("@");
    EXPECT_THROW(lexer.getNextToken(), std::invalid_argument);
}

TEST(TokenLexer, RejectsMalformedNumericLiterals) {
    Lexer lexer("1e+");
    EXPECT_THROW(lexer.getNextToken(), std::invalid_argument);
}
