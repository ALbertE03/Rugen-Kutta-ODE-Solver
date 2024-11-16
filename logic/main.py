from astAL import Expression
from lexer import Lexer, TOKEN_PATTERNS, CONSTANTS, Token
from parser import Parser
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple,List


lexer = Lexer(TOKEN_PATTERNS, CONSTANTS)
tokens: list[Token] = lexer.tokenize(function)
parser = Parser()
ast: Expression = parser.make_ast(tokens)
print(ast)


def edo(ast: Expression, vars: dict)->Expression:
    variables = {"e": 2.718281828459045, "pi": 3.14}
    for key, value in vars.items():
        variables[key] = value
    return ast.eval(variables)



def RungeKutta(x_point:int, y_point:int, h:int)->Tuple[List[float|int],List[float|int]]:
    
    X_right = np.arange(x_point,10,h)
    
    y_right = np.zeros(len(X_right))
    
    y_right[0]=y_point

    #### Runge-Kutta 
    for i in range(len(X_right)-1): 
        k1 =  edo(ast,{'x': X_right[i], 'y': y_right[i]})
        k2 =  edo(ast,{'x': X_right[i] + h / 2, 'y': y_right[i] + k1 / 2})
        k3 =  edo(ast,{'x': X_right[i] + h / 2, 'y': y_right[i] + k2 / 2})
        k4 =  edo(ast,{'x': X_right[i] + h, 'y': y_right[i] + k3})
        y_right[i+1] = (y_right[i] + h*(1/6 * k1 + 1/3 * k2 + 1/3 * k3 + 1/6 * k4))

    # si retorna nan 0 inf es que dividió por 0 o esta haciendo cosas en lugares indefinidos  
    return X_right,y_right



RungeKutta(x_point,y_point,h)



