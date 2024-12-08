import numpy as np

def maquina_turing_discreta(func, dominio, maximizar=True):
    """
    Simula una máquina de Turing para funciones discretas.
    :param func: Función a optimizar.
    :param dominio: Lista o rango de valores discretos de x.
    :param maximizar: True para maximizar, False para minimizar.
    :return: (x_opt, f_opt) El valor óptimo de x y su correspondiente f(x).
    """
    x_opt = None
    f_opt = float('-inf') if maximizar else float('inf')

    for x in dominio:
        f_x = func(x)
        if (maximizar and f_x > f_opt) or (not maximizar and f_x < f_opt):
            f_opt = f_x
            x_opt = x

    return x_opt, f_opt


def maquina_turing_continua(func, a, b, delta_x, maximizar=True, refinamiento=False, iteraciones=1):
    """
    Simula una máquina de Turing para funciones continuas.
    :param func: Función a optimizar.
    :param a: Límite inferior del dominio.
    :param b: Límite superior del dominio.
    :param delta_x: Tamaño de paso para discretización.
    :param maximizar: True para maximizar, False para minimizar.
    :param refinamiento: True para usar refinamientos adicionales.
    :param iteraciones: Número de refinamientos.
    :return: (x_opt, f_opt) El valor óptimo aproximado de x y su correspondiente f(x).
    """
    for _ in range(iteraciones):
        x_values = np.arange(a, b + delta_x, delta_x)
        x_opt, f_opt = maquina_turing_discreta(func, x_values, maximizar)

        # Refinar el intervalo
        if refinamiento:
            a = max(a, x_opt - delta_x)
            b = min(b, x_opt + delta_x)
            delta_x /= 2  # Reducir el tamaño de paso

    return x_opt, f_opt


def maquina_turing(func, dominio, tipo="discreto", **kwargs):
    """
    Controlador para ejecutar la máquina de Turing según el tipo de dominio.
    :param func: Función a optimizar.
    :param dominio: Dominio discreto (lista/rango) o continuo (tupla con límites).
    :param tipo: "discreto" o "continuo".
    :param kwargs: Parámetros adicionales para funciones discretas o continuas.
    :return: (x_opt, f_opt) El valor óptimo de x y su correspondiente f(x).
    """
    if tipo == "discreto":
        return maquina_turing_discreta(func, dominio, **kwargs)
    elif tipo == "continuo":
        if not isinstance(dominio, tuple) or len(dominio) != 2:
            raise ValueError("El dominio para 'continuo' debe ser una tupla (a, b).")
        a, b = dominio
        delta_x = kwargs.get("delta_x", 0.1)
        refinamiento = kwargs.get("refinamiento", False)
        iteraciones = kwargs.get("iteraciones", 1)
        maximizar = kwargs.get("maximizar", True)
        return maquina_turing_continua(func, a, b, delta_x, maximizar, refinamiento, iteraciones)
    else:
        raise ValueError("El parámetro 'tipo' debe ser 'discreto' o 'continuo'.")


# Ejemplo de uso
if __name__ == "__main__":
    # Función discreta: f(x) = -x^2 + 4x
    def f_discreta(x):
        return -x**2 + 4*x

    # Función continua: f(x) = -x^2 + 4x
    def f_continua(x):
        return -x**2 + 4*x

    # Caso discreto
    dominio_discreto = range(-10, 11)  # Dominio discreto: x = -10, -9, ..., 10
    x_opt_discreta, f_opt_discreta = maquina_turing(f_discreta, dominio_discreto, tipo="discreto", maximizar=True)
    print(f"Óptimo discreto: x = {x_opt_discreta}, f(x) = {f_opt_discreta}")

    # Caso continuo
    dominio_continuo = (0, 4)  # Dominio continuo: [0, 4]
    x_opt_continua, f_opt_continua = maquina_turing(f_continua, dominio_continuo, tipo="continuo",
                                                    delta_x=0.1, refinamiento=True, iteraciones=3, maximizar=True)
    print(f"Óptimo continuo: x = {x_opt_continua:.4f}, f(x) = {f_opt_continua:.4f}")
