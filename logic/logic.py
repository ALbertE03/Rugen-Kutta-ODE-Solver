from astAL import Expression
from lexer import Lexer, TOKEN_PATTERNS, CONSTANTS, Token
from parser import Parser
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from error import *
from matplotlib.quiver import Quiver


class RungeKutta:
    def __init__(self, x0: int, y0: int, xf: int, h: float, function: str):
        self.x0 = x0
        self.y0 = y0
        self.h = h
        self.xf = xf
        try:
            self.lexer = Lexer(TOKEN_PATTERNS, CONSTANTS)
            self.tokens: list[Token] = self.lexer.tokenize(function)
            self.parser = Parser()
            self.ast: Expression = self.parser.make_ast(self.tokens)
        except Parentesis_Error as e:
            raise Parentesis_Error(e.mensaje)

    def edo(self, vars: dict) -> float:
        variables = {"e": 2.718281828459045, "pi": 3.141592653589793}
        for key, value in vars.items():
            variables[key] = value
        return self.ast.eval(variables)

    def solver(self) -> Tuple[List[float], List[float]]:
        try:
            X_right = np.arange(self.x0, self.xf, self.h)

            y_right = np.zeros(len(X_right))

            y_right[0] = self.y0

            #### Runge-Kutta
            for i in range(len(X_right) - 1):
                k1 = self.edo({"x": X_right[i], "y": y_right[i]})
                k2 = self.edo({"x": X_right[i] + self.h / 2, "y": y_right[i] + k1 / 2})
                k3 = self.edo({"x": X_right[i] + self.h / 2, "y": y_right[i] + k2 / 2})
                k4 = self.edo({"x": X_right[i] + self.h, "y": y_right[i] + k3})
                y_right[i + 1] = y_right[i] + self.h * (
                    k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
                )
            # si retorna nan o inf es que dividió por 0 o esta haciendo cosas en lugares indefinidos
            return X_right, y_right
        except:
            raise RK_Error()

    def isoclinas(self) -> Quiver:
        X_right = np.arange(self.x0, self.xf, self.h)
        x_values = np.linspace(-10, 10, 20)
        y_values = np.linspace(-10, 10, 20)
        X, Y = np.meshgrid(x_values, y_values)
        U = 1
        V = self.edo({"x": X, "y": Y})
        return plt.quiver(X, Y, U, V, color="lightgray")


h = float(input())
x0 = int(input())
y0 = int(input())
xf = int(input())
f = input()
rk = RungeKutta(x0, y0, xf, h, f)

print(rk.ast)
