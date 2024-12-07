from astAL import Expression
from lexer import Lexer, TOKEN_PATTERNS, CONSTANTS, Token
from parser import Parser
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List,Callable

import numpy as np
from typing import List, Tuple

def edo(ast: Expression, vars: dict) -> float:
    variables = {"e": 2.718281828459045, "pi": 3.141592653589793}
    for key, value in vars.items():
        variables[key] = value
    return ast.eval(variables)

def rk4_step(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h / 2, y + k1 / 2)
    k3 = h * f(t + h / 2, y + k2 / 2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def rk4_step_backward(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t - h / 2, y - k1 / 2)
    k3 = h * f(t - h / 2, y - k2 / 2)
    k4 = h * f(t - h, y - k3)
    return y - (k1 + 2*k2 + 2*k3 + k4) / 6

def RungeKutta(x0: int, y0: float, xf: int, h: float,function:Callable) -> Tuple[List[float], List[float]]:
    lexer = Lexer(TOKEN_PATTERNS, CONSTANTS)
    tokens: list[Token] = lexer.tokenize(function)
    parser = Parser()
    ast: Expression = parser.make_ast(tokens)

    # Resolver hacia adelante
    N_forward = int((xf - x0) / h) + 1
    t_values_forward = np.linspace(x0, xf, N_forward)
    v_values_forward = np.zeros(N_forward)

    v_values_forward[0] = y0

    for n in range(1, N_forward):
        x_n = t_values_forward[n-1]
        f_n = v_values_forward[n-1]

        v_values_forward[n] = rk4_step(lambda t, y: edo(ast, {'x': t, 'y': y}), x_n, f_n, h)

    # Resolver hacia atrás
    N_backward = int((xf - x0) / h) + 1
    t_values_backward = np.linspace(x0 - (N_backward - 1) * h, x0, N_backward)
    v_values_backward = np.zeros(N_backward)

    v_values_backward[0] = y0

    for n in range(1, N_backward):
        x_n = t_values_backward[n-1]
        f_n = v_values_backward[n-1]

        v_values_backward[n] = rk4_step_backward(lambda t, y: edo(ast, {'x': t, 'y': y}), x_n, f_n, -h)

    # Invertir los resultados hacia atrás para combinarlos
    t_values_backward = t_values_backward[::-1]
    v_values_backward = v_values_backward[::-1]

    # Combinar los resultados (evitar duplicados en el punto medio)
    t_combined = np.concatenate((t_values_backward[:-1], t_values_forward))
    v_combined = np.concatenate((v_values_backward[:-1], v_values_forward))

    return t_combined.tolist(), v_combined.tolist()

# Ejemplo de uso
if __name__ == "__main__":
    # Definir la función a resolver aquí
    function = "x - y"  # Por ejemplo: dy/dt = x - y

    # Parámetros iniciales
    x0 = 1      # Valor inicial de x
    y0 = 0      # Valor inicial de y
    xf = 10     # Valor final de x
    h = 0.1     # Tamaño del paso

    # Resolver la EDO
    t_values, v_values = RungeKutta(x0, y0, xf, h,function)
    print(t_values)
    plt.plot(t_values[1:], v_values[1:], label='Solución RK4', color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solución de la EDO usando RK4')
    plt.grid()
    plt.legend()
    
    plt.show()