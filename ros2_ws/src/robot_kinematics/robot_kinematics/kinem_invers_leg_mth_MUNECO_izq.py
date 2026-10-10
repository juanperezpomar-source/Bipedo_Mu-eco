"""
Cinemática inversa de la pata del robot bípedo.
NUEVO MODELO: 5 parámetros (L1 a L5), 4 matrices DH.
"""

import numpy as np

# Dimensiones físicas del muñeco (mm)
L1, L2, L3, L4, L5 = 51.06, 78.37, 54.2, 138.5, 160.52

q1_deg, q2_deg, q3_deg = 0, 0, 0   # ángulos de prueba

# Límites articulares [°]
q1_min, q1_max = 0, 90
q2_min, q2_max = -90, 90
q3_min, q3_max = -90, 90


def dh_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]])

def fk_T(q1, q2, q3, L1, L2, L3, L4, L5):
    # Nueva Cinemática Directa basada en tu modelo
    A01 = dh_matrix(0,  d=-L1, a=L2, alpha=-np.pi/2)
    A12 = dh_matrix(q1, d=0,   a=L3, alpha=np.pi/2)
    A23 = dh_matrix(q2, d=0,   a=L4, alpha=np.pi)
    A34 = dh_matrix(q3, d=0,   a=L5, alpha=0)
    return A01 @ A12 @ A23 @ A34


def cinematica_inversa_pata_mth(T, L1, L2, L3, L4, L5):
    """
    Calcula q1, q2, q3 [rad] leyendo directamente las celdas de T.
    Adaptado al nuevo modelo de 5 eslabones.
    """
    nx, ny, nz = T[0, 0], T[1, 0], T[2, 0]
    oy = T[1, 1]
    ax, az = T[0, 2], T[2, 2]
    px, py, pz = T[0, 3], T[1, 3], T[2, 3]

    # --- 1. Calcular q1 (Rotación de la cadera) ---
    q1 = np.arctan2(-ax, -az)
    c1, s1 = np.cos(q1), np.sin(q1)

    # --- 2. Proyección espacial (Px se mantiene idéntico) ---
    Px = c1*(px - L2) - s1*(pz + L1)

    # --- 3. Calcular q2 usando el desacoplo geométrico ---
    # Se reemplaza L6 por L5, L5 por L4, y L4 por L3
    s2 = (py - L5*ny) / L4
    c2 = (Px - L3 + L5*oy) / L4
    q2 = np.arctan2(s2, c2)

    # --- 4. Calcular q3 mediante la relación q23 ---
    q23 = np.arctan2(ny, -oy)
    q3 = q2 - q23
    
    # Normalizamos q3 a un rango de (-180°, 180°]
    q3 = np.arctan2(np.sin(q3), np.cos(q3))

    # --- 5. Verificación de si la coordenada es alcanzable ---
    alcanzable = np.max(np.abs(fk_T(q1, q2, q3, L1, L2, L3, L4, L5) - T)) < 1e-6
    
    return q1, q2, q3, alcanzable


def verificar_con_mth(q1, q2, q3, x_obj=None, y_obj=None, z_obj=None,
                      L1=L1, L2=L2, L3=L3, L4=L4, L5=L5, tol=1e-6):
    """
    Verifica, con el método de la matriz de transformación homogénea,
    la solución (q1, q2, q3) [rad] obtenida por otro método de
    cinemática inversa (algebraico, geométrico, Newton, gradiente o
    desacople).

    1. Arma Tdes = fk_T(q1, q2, q3)  (cinemática directa).
    2. Le pasa Tdes al método MTH, que intenta recuperar los mismos
       ángulos leyendo únicamente la matriz.
    3. Compara los ángulos recuperados contra los originales y,
       si se dan x_obj/y_obj/z_obj, la posición de Tdes contra el
       punto que se le había pedido al método original.

    Retorna un diccionario:
        T         -> matriz Tdes (4x4)
        q1r/q2r/q3r -> ángulos [rad] recuperados por el método MTH
        q_error   -> error angular máximo [rad] entre q original y recuperado
        pos_error -> error de posición entre Tdes y (x_obj,y_obj,z_obj),
                     o None si no se dio el objetivo
        ok        -> True si fue alcanzable y q_error < tol
    """
    Tdes = fk_T(q1, q2, q3, L1, L2, L3, L4, L5)
    q1r, q2r, q3r, alcanzable = cinematica_inversa_pata_mth(
        Tdes, L1, L2, L3, L4, L5)

    if not alcanzable:
        return {
            'T': Tdes,
            'q1r': None, 'q2r': None, 'q3r': None,
            'q_error': None, 'pos_error': None,
            'ok': False,
        }

    def dif_angular(a, b):
        return abs(np.arctan2(np.sin(a - b), np.cos(a - b)))

    q_error = max(
        dif_angular(q1, q1r),
        dif_angular(q2, q2r),
        dif_angular(q3, q3r),
    )

    pos_error = None
    if x_obj is not None:
        pos_error = float(np.linalg.norm(
            Tdes[:3, 3] - np.array([x_obj, y_obj, z_obj])
        ))

    return {
        'T': Tdes,
        'q1r': q1r, 'q2r': q2r, 'q3r': q3r,
        'q_error': float(q_error),
        'pos_error': pos_error,
        'ok': bool(q_error < tol),
    }


# ==========================================
# EJECUCIÓN Y PRUEBA DE ESCRITORIO
# ==========================================
if __name__ == "__main__":
    qd = np.radians([q1_deg, q2_deg, q3_deg])
    Tdes = fk_T(*qd, L1, L2, L3, L4, L5)

    q1, q2, q3, alcanzable = cinematica_inversa_pata_mth(Tdes, L1, L2, L3, L4, L5)

    print(f"{'='*50}")
    print(f"q deseados de prueba : q1 = {q1_deg}°  q2 = {q2_deg}°  q3 = {q3_deg}°")
    print(f"Matriz Tdes generada:\n{np.round(Tdes, 4)}")
    print(f"{'='*50}")

    if not alcanzable:
        print("No alcanzable (La matriz solicitada está fuera del espacio de trabajo)")
    else:
        q1d, q2d, q3d = np.degrees([q1, q2, q3])
        print(f"q resueltos por IK   : q1 = {q1d:.2f}°  q2 = {q2d:.2f}°  q3 = {q3d:.2f}°")
        print(f"Error residual de T  : {np.max(np.abs(fk_T(q1, q2, q3, L1, L2, L3, L4, L5) - Tdes)):.2e}")
    
        for nombre, v, mn, mx in (("q1", q1d, q1_min, q1_max),
                                  ("q2", q2d, q2_min, q2_max),
                                  ("q3", q3d, q3_min, q3_max)):
            if not mn <= v <= mx:
                print(f"Advertencia: {nombre} ({v:.2f}°) fuera de rango mecánico [{mn}°, {mx}°]")
