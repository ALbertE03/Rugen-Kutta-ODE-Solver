class RK_Error(Exception):
    def __init__(self, mensaje="Función no válida"):
        self.mensaje = mensaje
        super().__init__(self.mensaje)

class Parentesis_Error(Exception):
    def __init__(self, mensaje="parentesis mal colocados"):
        self.mensaje = mensaje
        super().__init__(self.mensaje)