from logic.lexer import Token

import numpy as np


class Term:
    def __init__(self, token: Token) -> None:
        self.term: Token = token

    def eval(self, variables) -> Token:
        if self.term.lex in variables:
            return variables[self.term.lex]
        return float(self.term.lex)

    def __str__(self) -> str:
        return f"{self.term.lex}"


class Expression:

    def __init__(self, left, right) -> None:
        self.left = left
        self.right = right

    def eval(self):
        pass


class Plus(Expression):

    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables: dict):
        return self.left.eval(variables) + self.right.eval(variables)

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


class Minus(Expression):

    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables):
        return self.left.eval(variables) - self.right.eval(variables)

    def __str__(self) -> str:
        return f"({self.left} - {self.right})"


class Divide(Expression):

    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables):
        return self.left.eval(variables) / self.right.eval(variables)

    def __str__(self) -> str:
        return f"({self.left} / {self.right})"


class Times(Expression):

    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables):
        return self.left.eval(variables) * self.right.eval(variables)

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"


class Power(Expression):

    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables):
        return self.left.eval(variables) ** self.right.eval(variables)

    def __str__(self) -> str:
        return f"({self.left} ^ {self.right})"


class Sen(Expression):
    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables):
        return np.sin(self.left.eval(variables))

    def __str__(self) -> str:
        return f"sin({self.left})"


class Cos(Expression):
    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables):
        return np.cos(self.left.eval(variables))

    def __str__(self) -> str:
        return f"cos({self.left})"


class Tan(Expression):
    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables):
        return np.tan(self.left.eval(variables))

    def __str__(self) -> str:
        return f"tan({self.left})"


class Cot(Expression):
    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables):
        return np.cos(self.left.eval(variables)) / np.sin(self.left.eval(variables))

    def __str__(self) -> str:
        return f"cot({self.left})"


class Ln(Expression):
    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables):
        return np.log(self.left.eval(variables))

    def __str__(self) -> str:
        return f"ln({self.left})"


class Arctan(Expression):
    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables):
        return np.arctan(self.left.eval(variables))

    def __str__(self):
        return f"arctan({self.left})"


class Arcsin(Expression):

    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables):
        return np.arcsin(self.left.eval(variables))

    def __str__(self):
        return f"arcsin({self.left})"


class Arccos(Expression):

    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables):
        return np.arccos(self.left.eval(variables))

    def __str__(self):
        return f"arccos({self.left})"


class Log(Expression):
    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables):
        return np.log10(self.left.eval(variables))

    def __str__(self) -> str:
        return f"log({self.left})"
