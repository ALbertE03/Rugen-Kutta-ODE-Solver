from logic.lexer import Token
import warnings
import numpy as np
from logic.error import LnLog

warnings.simplefilter("error", RuntimeWarning)


class Term:
    def __init__(self, token: Token) -> None:
        self.term: Token = token

    def eval(self, variables) -> Token:
        if self.term.lex in variables:
            return variables[self.term.lex]
        return float(self.term.lex)

    def __str__(self) -> str:
        return f"{self.term.lex}"

    def __bool__(self) -> bool:
        return self.term is not None


class Expression:

    def __init__(self, left, right) -> None:
        self.left = left
        self.right = right

    def eval(self) -> float:
        pass

    def __bool__(self) -> bool:
        return self.left is not None and self.right is not None


class Plus(Expression):

    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables: dict) -> float:
        return self.left.eval(variables) + self.right.eval(variables)

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"

    def __bool__(self) -> bool:
        return super().__bool__()


class Minus(Expression):

    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables) -> float:
        return self.left.eval(variables) - self.right.eval(variables)

    def __str__(self) -> str:
        return f"({self.left} - {self.right})"

    def __bool__(self) -> bool:
        return super().__bool__()


class Negative(Expression):
    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables: dict) -> float:
        return -1 * self.right.eval(variables)

    def __str__(self) -> str:
        return f"-{self.right}"

    def __bool__(self) -> bool:
        return super().__bool__()


class Divide(Expression):

    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables) -> float:
        try:
            return self.left.eval(variables) / self.right.eval(variables)
        except ZeroDivisionError as e:
            raise ZeroDivisionError()
        except RuntimeWarning as e:
            raise ZeroDivisionError()

    def __str__(self) -> str:
        return f"({self.left} / {self.right})"

    def __bool__(self) -> bool:
        return super().__bool__()


class Times(Expression):

    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables) -> bool:
        return self.left.eval(variables) * self.right.eval(variables)

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"

    def __bool__(self) -> bool:
        return super().__bool__()


class Power(Expression):

    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables) -> float:
        try:
            return self.left.eval(variables) ** self.right.eval(variables)
        except RuntimeWarning as e:
            raise RuntimeWarning()

    def __str__(self) -> str:
        return f"({self.left} ^ {self.right})"

    def __bool__(self) -> bool:
        return super().__bool__()


class Sen(Expression):
    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables) -> float:
        return np.sin(self.left.eval(variables))

    def __str__(self) -> str:
        return f"sin({self.left})"

    def __bool__(self) -> bool:
        return super().__bool__()


class Cos(Expression):
    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables) -> float:
        return np.cos(self.left.eval(variables))

    def __str__(self) -> str:
        return f"cos({self.left})"

    def __bool__(self) -> bool:
        return super().__bool__()


class Tan(Expression):
    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables) -> float:
        return np.tan(self.left.eval(variables))

    def __str__(self) -> str:
        return f"tan({self.left})"

    def __bool__(self) -> bool:
        return super().__bool__()


class Cot(Expression):
    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables) -> float:
        return np.cos(self.left.eval(variables)) / np.sin(self.left.eval(variables))

    def __str__(self) -> str:
        return f"cot({self.left})"

    def __bool__(self) -> bool:
        return super().__bool__()


class Ln(Expression):
    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables) -> float:
        try:
            return np.log(self.left.eval(variables))
        except:
            raise LnLog()

    def __str__(self) -> str:
        return f"ln({self.left})"

    def __bool__(self) -> bool:
        return super().__bool__()


class Arctan(Expression):
    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables) -> float:
        return np.arctan(self.left.eval(variables))

    def __str__(self) -> str:
        return f"arctan({self.left})"

    def __bool__(self) -> bool:
        return super().__bool__()


class Arcsin(Expression):

    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables) -> float:
        return np.arcsin(self.left.eval(variables))

    def __str__(self) -> str:
        return f"arcsin({self.left})"

    def __bool__(self) -> bool:
        return super().__bool__()


class Arccos(Expression):

    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables) -> float:
        return np.arccos(self.left.eval(variables))

    def __str__(self) -> str:
        return f"arccos({self.left})"

    def __bool__(self) -> bool:
        return super().__bool__()


class Log(Expression):
    def __init__(self, left, right) -> None:
        super().__init__(left, right)

    def eval(self, variables) -> float:
        try:
            return np.log10(self.left.eval(variables))
        except:
            raise LnLog()

    def __str__(self) -> str:
        return f"log({self.left})"

    def __bool__(self) -> bool:
        return super().__bool__()
