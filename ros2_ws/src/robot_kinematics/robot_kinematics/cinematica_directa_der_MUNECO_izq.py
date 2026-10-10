"""
Cinemática directa del robot bípedo.

Implementación manual mediante Denavit-Hartenberg.
Los ángulos de entrada se manejan en grados.
Modelo viejo: 5 longitudes (L1..L5), sin articulación fantasma ni L6.

Tabla DH pierna DERECHA:

        theta   d    a     alpha
  0A1    0     L1    L2     90°
  1A2    q1    0     L3    -90°
  2A3    q2    0     L4     180°
  3A4    q3    0     L5     0

Tabla DH pierna IZQUIERDA (espejo en Z: cambian L1 y los alpha de 0A1 y 1A2):

        theta   d    a     alpha
  0A1    0    -L1    L2    -90°
  1A2    q1    0     L3     90°
  2A3    q2    0     L4     180°
  3A4    q3    0     L5     0
"""

import numpy as np


# ----------------------------------------------------------------------
# Parámetros geométricos
# ----------------------------------------------------------------------

L1 = 51.06
L2 = 78.37
L3 = 54.2
L4 = 138.5
L5 = 160.52

# ----------------------------------------------------------------------
# Matriz de transformación DH
# ----------------------------------------------------------------------

def dh_matrix(theta, d, a, alpha):
    """
    Calcula la matriz homogénea 4x4
    utilizando la convención DH estándar.

    Parámetros:
        theta : ángulo articular [rad]
        d     : desplazamiento [m]
        a     : longitud del eslabón [m]
        alpha : ángulo de torsión [rad]
    """

    ct = np.cos(theta)
    st = np.sin(theta)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1]
    ])


# ----------------------------------------------------------------------
# CINEMÁTICA PIERNA DERECHA
# ----------------------------------------------------------------------

def forward_kinematics_right(q):
    """
    Calcula las transformaciones homogéneas de la pierna derecha.

    q:
        [hip_roll, hip_pitch, knee] en grados

    Retorna:
        T01, T02, T03, T04
    """

    if len(q) != 3:
        raise ValueError(
            "La pierna derecha debe recibir exactamente 3 ángulos."
        )

    q1, q2, q3 = np.radians(q)

    # --------------------------------------------------------------
    # Transformaciones DH
    # --------------------------------------------------------------

    A01 = dh_matrix(
        0,
        d=L1,
        a=L2,
        alpha=np.pi / 2
    )

    A12 = dh_matrix(          # hip_roll
        q1,
        d=0,
        a=L3,
        alpha=-np.pi / 2
    )

    A23 = dh_matrix(          # hip_pitch
        q2,
        d=0,
        a=L4,
        alpha=np.pi
    )

    A34 = dh_matrix(          # knee
        q3,
        d=0,
        a=L5,
        alpha=0
    )

    # --------------------------------------------------------------
    # Transformaciones acumuladas
    # --------------------------------------------------------------

    T01 = A01
    T02 = T01 @ A12
    T03 = T02 @ A23
    T04 = T03 @ A34

    return T01, T02, T03, T04


# ----------------------------------------------------------------------
# CINEMÁTICA PIERNA IZQUIERDA
# ----------------------------------------------------------------------

def forward_kinematics_left(q):
    """
    Calcula las transformaciones homogéneas de la pierna izquierda.

    q:
        [hip_roll, hip_pitch, knee] en grados

    Retorna:
        T01, T02, T03, T04
    """

    if len(q) != 3:
        raise ValueError(
            "La pierna izquierda debe recibir exactamente 3 ángulos."
        )

    q1, q2, q3 = np.radians(q)

    # --------------------------------------------------------------
    # Transformaciones DH (mismas L2..L5 y mismos alpha en 2A3 y 3A4
    # que la pierna derecha; solo se refleja el signo de L1 y los
    # alpha del tramo cadera, 0A1 y 1A2)
    # --------------------------------------------------------------

    A01 = dh_matrix(
        0,
        d=-L1,
        a=L2,
        alpha=-np.pi / 2
    )

    A12 = dh_matrix(          # hip_roll
        q1,
        d=0,
        a=L3,
        alpha=np.pi / 2
    )

    A23 = dh_matrix(          # hip_pitch
        q2,
        d=0,
        a=L4,
        alpha=np.pi
    )

    A34 = dh_matrix(          # knee
        q3,
        d=0,
        a=L5,
        alpha=0
    )

    # --------------------------------------------------------------
    # Transformaciones acumuladas
    # --------------------------------------------------------------

    T01 = A01
    T02 = T01 @ A12
    T03 = T02 @ A23
    T04 = T03 @ A34

    return T01, T02, T03, T04


# ----------------------------------------------------------------------
# CINEMÁTICA COMPLETA DEL BÍPEDO
# ----------------------------------------------------------------------

def forward_kinematics_biped(q_right, q_left):
    """
    Calcula todas las transformaciones de ambas piernas.

    Retorna:

        {
            "right": (T01, T02, T03, T04),
            "left":  (T01, T02, T03, T04)
        }
    """

    right = forward_kinematics_right(q_right)
    left = forward_kinematics_left(q_left)

    return {
        "right": right,
        "left": left
    }


# ----------------------------------------------------------------------
# Posición
# ----------------------------------------------------------------------

def get_position(T):
    """
    Extrae la posición XYZ de una MTH.
    """

    return T[0:3, 3]


# ----------------------------------------------------------------------
# Rotación
# ----------------------------------------------------------------------

def get_rotation(T):
    """
    Extrae la matriz de rotación 3x3.
    """

    return T[0:3, 0:3]


# ----------------------------------------------------------------------
# Formato para mostrar en la interfaz
# ----------------------------------------------------------------------

def format_matrix(T):
    """
    Convierte una matriz 4x4 en texto para mostrarla en Tkinter.
    """

    return (
        f"{T[0,0]:8.4f}  {T[0,1]:8.4f}  {T[0,2]:8.4f}  {T[0,3]:8.4f}\n"
        f"{T[1,0]:8.4f}  {T[1,1]:8.4f}  {T[1,2]:8.4f}  {T[1,3]:8.4f}\n"
        f"{T[2,0]:8.4f}  {T[2,1]:8.4f}  {T[2,2]:8.4f}  {T[2,3]:8.4f}\n"
        f"{T[3,0]:8.4f}  {T[3,1]:8.4f}  {T[3,2]:8.4f}  {T[3,3]:8.4f}"
    )
