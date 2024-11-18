from lexer import Token, TokenType
from astAL import (
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
}


class Parser:

    def make_ast(self, tokens: list[Token]) -> Expression:
        if len(tokens) == 1:
            return Term(tokens[0])
        self.is_parenthesis_balance(tokens)

        expression: Expression = self.search_expression(tokens, self.is_term)
        if expression == None:
            expression: Expression = self.search_expression(tokens, self.is_factor)
        if expression == None:
            expression: Expression = self.search_expression(tokens, self.is_power)
        return expression

    def search_expression(self, tokens: list[Token], func) -> Expression:
        balance = 0
        for i in range(len(tokens)):

            if tokens[i].token_type == TokenType.LEFT_PARENTHESIS:
                balance += 1
            elif tokens[i].token_type == TokenType.RIGHT_PARENTHESIS:
                balance -= 1

            if balance == 0 and func(tokens[i]):
                return token_to_class[tokens[i].token_type](
                    self.make_ast(tokens[:i]), self.make_ast(tokens[i + 1 :])
                )
        if (
            tokens[0].token_type == TokenType.LEFT_PARENTHESIS
            and tokens[-1].token_type == TokenType.RIGHT_PARENTHESIS
        ):
            return self.make_ast(tokens[1:-1])
        if (
            tokens[0].token_type == TokenType.SEN
            or tokens[0].token_type == TokenType.COS
            or tokens[0].token_type == TokenType.TAN
            or tokens[0].token_type == TokenType.COT
            or tokens[0].token_type == TokenType.LN
        ):
            return token_to_class[tokens[0].token_type](self.make_ast(tokens[1:]), None)

    def is_term(self, token: Token) -> bool:
        return token.token_type == TokenType.PLUS or token.token_type == TokenType.MINUS

    def is_factor(self, token: Token) -> bool:
        return (
            token.token_type == TokenType.TIMES or token.token_type == TokenType.DIVIDE
        )

    def is_power(self, token: Token) -> bool:
        return token.token_type == TokenType.POWER

    def is_parenthesis_balance(self, tokens: list[Token]) -> None:
        balance = 0
        for token in tokens:
            if token.token_type == TokenType.LEFT_PARENTHESIS:
                balance += 1
            elif token.token_type == TokenType.RIGHT_PARENTHESIS:
                balance -= 1
        if balance != 0:
            raise "Ha habido un error con los parentesis"
