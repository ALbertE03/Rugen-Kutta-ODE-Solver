class RK_Error(Exception):
    def __init__(self, mensaje="Función no válida"):
        self.mensaje = mensaje
        super().__init__(self.mensaje)


class Parentesis_Error(Exception):
    def __init__(self, mensaje="parentesis mal colocados"):
        self.mensaje = mensaje
        super().__init__(self.mensaje)


class Inf(Exception):
    def __init__(
        self,
        mensaje="""
No se puede buscar solución en el intervalo dado, 
debido a que no está definida en uno o más puntos de este. 
Porfavor introduzca otro intervalo.""",
    ):
        self.mensaje = mensaje
        super().__init__(self.mensaje)


class TokenError(Exception):
    def __init__(self, mensaje="Existe un token inválido en la función."):
        self.mensaje = mensaje
        super().__init__(self.mensaje)


class SEL(Exception):
    def __init__(
        self,
        mensaje="""
    escriba 2 escuaciones
    """,
    ):
        self.mensaje = mensaje
        super().__init__(self.mensaje)


class LnLog(Exception):

    def __init__(
        self,
        mensaje="""En este intervalo el Logaritmo no se encuentra definido
    """,
    ):
        self.mensaje = mensaje
        super().__init__(self.mensaje)
