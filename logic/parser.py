from logic.lexer import Token, TokenType, Lexer, CONSTANTS, TOKEN_PATTERNS
from logic.error import Parentesis_Error
from logic.astAL import (
    Expression,
    Term,
    Plus,
    Minus,
    Power,
    Divide,
    Times,
    Sen,
    Cos,
    Tan,
    Cot,
    Log,
    Ln,
    Arctan,
    Arcsin,
    Arccos,
    Negative,
)


token_to_class = {
    TokenType.PLUS: Plus,
    TokenType.MINUS: Minus,
    TokenType.POWER: Power,
    TokenType.DIVIDE: Divide,
    TokenType.TIMES: Times,
    TokenType.SEN: Sen,
    TokenType.COS: Cos,
    TokenType.TAN: Tan,
    TokenType.COT: Cot,
    TokenType.LOG: Log,
    TokenType.LN: Ln,
    TokenType.ARCTAN: Arctan,
    TokenType.ARCSIN: Arcsin,
    TokenType.ARCCOS: Arccos,
}


class Parser:

    def make_ast(self, tokens: list[Token]) -> Expression:
        if len(tokens) == 1:
            return Term(tokens[0])

        self.check_parenthesis_balance(tokens)
        expression = self.parse_expression(tokens)
        return expression

    def parse_expression(self, tokens: list[Token]) -> Expression:
        return self.parse_term(tokens)

    def parse_term(self, tokens: list[Token]) -> Expression:
        left = self.parse_factor(tokens)

        while tokens and tokens[0].token_type in {TokenType.PLUS, TokenType.MINUS}:
            operator = tokens.pop(0)
            right = self.parse_factor(tokens)
            left = token_to_class[operator.token_type](left, right)

        return left

    def parse_factor(self, tokens: list[Token]) -> Expression:
        left = self.parse_power(tokens)

        while tokens and self.is_factor(tokens[0]):
            operator = tokens.pop(0)
            right = self.parse_power(tokens)
            left = token_to_class[operator.token_type](left, right)

        return left

    def is_factor(self, token) -> bool:
        return token.token_type in {TokenType.TIMES, TokenType.DIVIDE}

    def parse_power(self, tokens: list[Token]) -> Expression:
        left = self.parse_unary(tokens)

        while tokens and tokens[0].token_type == TokenType.POWER:
            operator = tokens.pop(0)
            right = self.parse_unary(tokens)
            left = token_to_class[operator.token_type](left, right)

        return left

    def parse_unary(self, tokens: list[Token]) -> Expression:
        if tokens and tokens[0].token_type == TokenType.MINUS:
            operator = tokens.pop(0)
            operand = self.parse_unary(tokens)
            return Negative(None, operand)
        if tokens and self.is_unary_function(tokens[0]):
            operator = tokens.pop(0)
            operand = self.parse_unary(tokens)
            return token_to_class[operator.token_type](operand, None)

        return self.parse_number_var(tokens)

    def parse_number_var(self, tokens: list[Token]) -> Expression:

        if not tokens:
            raise ValueError()

        token = tokens.pop(0)

        if token.token_type in {
            TokenType.NUMBER,
            TokenType.IDENTIFIER,
            TokenType.E,
            TokenType.PI,
        }:
            return Term(token)

        if token.token_type == TokenType.LEFT_PARENTHESIS:
            expression = self.parse_expression(tokens)
            if not tokens or tokens.pop(0).token_type != TokenType.RIGHT_PARENTHESIS:
                raise Parentesis_Error("Paréntesis no balanceados.")
            return expression

        raise ValueError(f"Token inesperado: {token}")

    def is_unary_function(self, token: Token) -> bool:
        return token.token_type in {
            TokenType.SEN,
            TokenType.COS,
            TokenType.TAN,
            TokenType.COT,
            TokenType.LN,
            TokenType.LOG,
            TokenType.ARCTAN,
            TokenType.ARCSIN,
            TokenType.ARCCOS,
        }

    def check_parenthesis_balance(self, tokens: list[Token]) -> None:
        balance = 0
        for token in tokens:
            if token.token_type == TokenType.LEFT_PARENTHESIS:
                balance += 1
            elif token.token_type == TokenType.RIGHT_PARENTHESIS:
                balance -= 1
        if balance != 0:
            raise Parentesis_Error()
