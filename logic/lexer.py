from enum import Enum, auto
import re


class Token:
    def __init__(self, lex, token_type):
        self.lex: str = lex
        self.token_type: TokenType = token_type

    def __eq__(self, value: object) -> bool:
        return self.lex == value.lex and self.token_type == value.token_type

    def __str__(self):
        return f"Token( Lex:{self.lex}, Type:{self.token_type} )"

    def __repr__(self) -> str:
        return self.__str__()


class TokenType(Enum):

    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()

    # Arithmetic Operators
    PLUS = auto()
    MINUS = auto()
    TIMES = auto()
    DIVIDE = auto()
    POWER = auto()
    SEN = auto()
    COS = auto()
    TAN = auto()
    COT = auto()
    LOG = auto()
    LN = auto()

    # Relational Operators
    EQUAL = auto()

    # Punctuation
    LEFT_PARENTHESIS = auto()
    RIGHT_PARENTHESIS = auto()
    EOF = auto()

    # Mathematical constants
    PI = auto()
    E = auto()


CONSTANTS = {
    "PI": ("3.141592653589793", TokenType.NUMBER),
    "E": ("2.718281828459045", TokenType.NUMBER),
}


class TokenPattern:

    def __init__(self, regex_pattern: str, token_type: TokenType, follow: str = None):
        self.regex_pattern: re.Pattern[str] = re.compile(regex_pattern)
        self.token_type: TokenType = token_type
        self.follow: re.Pattern = re.compile(follow) if follow else None


_NOT_LETTER_OR_UNDERSCORE = "[^a-zA-Z_]"

TOKEN_PATTERNS = [
    TokenPattern("=", TokenType.EQUAL, "[^=>]"),
    TokenPattern('\\"([^\\"]*)\\"', TokenType.STRING),
    TokenPattern("(0\\.[0-9]+)|([1-9][0-9]*\\.?[0-9]*)|(0)", TokenType.NUMBER),
    TokenPattern("(\\^|\\*\\*)", TokenType.POWER),
    TokenPattern("\\+", TokenType.PLUS),
    TokenPattern("\\-", TokenType.MINUS),
    TokenPattern("\\*", TokenType.TIMES, "[^\\*]"),
    TokenPattern("\\/", TokenType.DIVIDE),
    TokenPattern("\\(", TokenType.LEFT_PARENTHESIS),
    TokenPattern("\\)", TokenType.RIGHT_PARENTHESIS),
    TokenPattern("pi", TokenType.PI, _NOT_LETTER_OR_UNDERSCORE),
    TokenPattern("e", TokenType.E, _NOT_LETTER_OR_UNDERSCORE),
    TokenPattern("sen", TokenType.SEN, _NOT_LETTER_OR_UNDERSCORE),
    TokenPattern("cos", TokenType.COS, _NOT_LETTER_OR_UNDERSCORE),
    TokenPattern("tan", TokenType.TAN, _NOT_LETTER_OR_UNDERSCORE),
    TokenPattern("cot", TokenType.COT, _NOT_LETTER_OR_UNDERSCORE),
    TokenPattern("ln", TokenType.LN, _NOT_LETTER_OR_UNDERSCORE),
    TokenPattern("log", TokenType.LOG, _NOT_LETTER_OR_UNDERSCORE),
    TokenPattern("[a-zA-Z_]([a-zA-Z_0-9])*", TokenType.IDENTIFIER),
]


class Lexer:

    def __init__(self, patterns: list[TokenPattern], constants: dict[str, str]) -> None:
        self.patterns: list[TokenPattern] = patterns
        self.constants: dict[str, str] = constants

    def tokenize(self, text: str) -> list[Token]:
        tokens: list[Token] = []

        i = 0
        while i < len(text):
            if text[i] == " ":
                i += 1
                continue

            for pattern in self.patterns:
                match = pattern.regex_pattern.match(text, i)

                if match:

                    if pattern.follow and match.end() < len(text):
                        if not pattern.follow.match(text, match.end()):
                            continue

                    i = match.end()
                    new_token = Token(
                        match.group(),
                        pattern.token_type,
                    )

                    if new_token.lex in self.constants:
                        new_token.token_type = self.constants[new_token.lex][1]
                        new_token.lex = self.constants[new_token.lex][0]

                    tokens.append(new_token)

                    break
            else:
                print("No se reconoce el token")

        token_expression: list[Token] = []
        we_are_in_the_right_member = False
        for token in tokens:
            if we_are_in_the_right_member:
                token_expression.append(token)
            else:
                if token.token_type == TokenType.EQUAL:
                    we_are_in_the_right_member = True
        if token_expression == []:
            token_expression = tokens

        response: list[Token] = []
        for token in token_expression:
            if response and add_times(response, token):
                response.append(Token(lex="*", token_type=TokenType.TIMES))
            response.append(token)

        return response


def add_times(response, token):
    return (
        response[-1].token_type == TokenType.IDENTIFIER
        and token.token_type == TokenType.NUMBER
        or response[-1].token_type == TokenType.NUMBER
        and token.token_type == TokenType.IDENTIFIER
        or response[-1].token_type == TokenType.IDENTIFIER
        and token.token_type == TokenType.LEFT_PARENTHESIS
        or response[-1].token_type == TokenType.NUMBER
        and token.token_type == TokenType.LEFT_PARENTHESIS
        or response[-1].token_type == TokenType.RIGHT_PARENTHESIS
        and token.token_type == TokenType.NUMBER
        or response[-1].token_type == TokenType.RIGHT_PARENTHESIS
        and token.token_type == TokenType.IDENTIFIER
    )
