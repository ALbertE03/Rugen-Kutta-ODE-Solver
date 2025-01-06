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
        balance = 0

        for fun in {
            self.is_term,
            self.is_factor,
            self.is_power,
            self.is_unary_function,
        }:
            for i, token in enumerate(tokens):

                if token.token_type == TokenType.LEFT_PARENTHESIS:
                    balance += 1
                elif token.token_type == TokenType.RIGHT_PARENTHESIS:
                    balance -= 1

                if balance == 0 and fun(token):
                    if token.token_type == TokenType.MINUS and i == 0:

                        return Negative(None, self.make_ast(tokens[1:]))

                    if (
                        self.is_term(token)
                        or self.is_factor(token)
                        or self.is_power(token)
                    ):
                        return token_to_class[token.token_type](
                            self.make_ast(tokens[:i]), self.make_ast(tokens[i + 1 :])
                        )

                    if self.is_unary_function(token):
                        return token_to_class[token.token_type](
                            self.make_ast(tokens[i + 2 : -1]), None
                        )

            if (
                tokens[0].token_type == TokenType.LEFT_PARENTHESIS
                and tokens[-1].token_type == TokenType.RIGHT_PARENTHESIS
            ):
                return self.make_ast(tokens[1:-1])

    def is_term(self, token: Token) -> bool:
        return token.token_type in {TokenType.PLUS, TokenType.MINUS}

    def is_factor(self, token: Token) -> bool:
        return token.token_type in {TokenType.TIMES, TokenType.DIVIDE}

    def is_power(self, token: Token) -> bool:
        return token.token_type == TokenType.POWER

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
