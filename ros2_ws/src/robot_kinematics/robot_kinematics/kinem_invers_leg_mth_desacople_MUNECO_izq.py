"""
Cinemática inversa de la pata del robot bípedo.

Implementación por desacople (premultiplicación por matrices inversas).
NUEVO MODELO: 5 parámetros (L1 a L5), 4 matrices DH (sin eslabón fantasma).
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
    # Cinemática directa actualizada (nuevo modelo)
    A01 = dh_matrix(0,  d=-L1, a=L2, alpha=-np.pi/2)
    A12 = dh_matrix(q1, d=0,   a=L3, alpha=np.pi/2)
    A23 = dh_matrix(q2, d=0,   a=L4, alpha=np.pi)
    A34 = dh_matrix(q3, d=0,   a=L5, alpha=0)
    return A01 @ A12 @ A23 @ A34


def cinematica_inversa_pata_des(T, L1, L2, L3, L4, L5, codo="arriba"):
    """
    Calcula q1, q2, q3 [rad] que producen la matriz T, despejando por
    desacople algebraico. Adaptado para 5 eslabones.
    """
    px, py, pz = T[0, 3], T[1, 3], T[2, 3]

    # --- 1. Calcular q1 ---
    # Ecuación: (pz+L1)*c1 + (px-L2)*s1 = C
    # En el nuevo modelo, al no haber desfase lateral (d) después de q1, C = 0.
    A = pz + L1
    B = px - L2
    C = 0  

    disc1 = A**2 + B**2 - C**2
    if disc1 < -1e-9:
        return None, None, None, False
    disc1 = max(disc1, 0.0)
    
    q1 = np.arctan2(B, A) - np.arctan2(np.sqrt(disc1), C)
    q1 = np.arctan2(np.sin(q1), np.cos(q1))    # envolver a (-180°, 180°]
    c1, s1 = np.cos(q1), np.sin(q1)

    # --- 2. Proyección f14, f24 ---
    # Columna de posición de F = (1A2)^-1 (0A1)^-1 Tdes
    f14 = c1*(px - L2) - s1*(pz + L1)
    f24 = py

    # --- 3. Calcular q2 (triángulo auxiliar) ---
    # Ecuación: Aq2 cos(q2) + Bq2 sin(q2) = Cq2
    # El offset base cambia de L4 a L3, y los eslabones móviles son L4 y L5
    Aq2 = f14 - L3
    Bq2 = f24
    Cq2 = (Aq2**2 + Bq2**2 + L4**2 - L5**2) / (2*L4)

    disc2 = Aq2**2 + Bq2**2 - Cq2**2
    if disc2 < -1e-9:
        return None, None, None, False
    disc2 = max(disc2, 0.0)

    signo = 1 if codo == "arriba" else -1
    q2 = np.arctan2(Bq2, Aq2) + signo*np.arctan2(np.sqrt(disc2), Cq2)
    q2 = np.arctan2(np.sin(q2), np.cos(q2))    # envolver
    c2, s2 = np.cos(q2), np.sin(q2)

    # --- 4. Calcular q3 ---
    # Se extrae de las ecuaciones de posición restantes sustituyendo L5 por L4
    q3 = np.arctan2(s2*Aq2 - c2*Bq2, c2*Aq2 + s2*Bq2 - L4)
    q3 = np.arctan2(np.sin(q3), np.cos(q3))

    # --- 5. Verificación ---
    alcanzable = np.max(np.abs(fk_T(q1, q2, q3, L1, L2, L3, L4, L5) - T)) < 1e-6
    return q1, q2, q3, alcanzable


def joint_limit_warnings(q1_deg, q2_deg, q3_deg, tol=1e-6):
    warnings = []
    for nombre, v, mn, mx in (("q1", q1_deg, q1_min, q1_max),
                              ("q2", q2_deg, q2_min, q2_max),
                              ("q3", q3_deg, q3_min, q3_max)):
        if not (mn - tol) <= v <= (mx + tol):
            warnings.append(f"{nombre} ({v:.2f}°) fuera de rango [{mn}°, {mx}°]")
    return warnings


def cinematica_inversa_pata_des_xyz(x, y, z, L1=L1, L2=L2, L3=L3, L4=L4,
                                    L5=L5, codo="arriba"):
    """
    Versión para un punto (x, y, z) [mm], que es lo que mandan los
    nodos ROS y la interfaz. El desacople solo usa la posición de T
    para despejar q1, q2, q3; la orientación no se puede elegir
    (3 GDL), así que el alcance se verifica solo por posición.

    Retorna (q1, q2, q3, alcanzable).
    """
    T = np.eye(4)
    T[:3, 3] = (x, y, z)

    q1, q2, q3, _ = cinematica_inversa_pata_des(
        T, L1, L2, L3, L4, L5, codo=codo)

    if q1 is None:
        return None, None, None, False

    p = fk_T(q1, q2, q3, L1, L2, L3, L4, L5)[:3, 3]
    alcanzable = bool(np.max(np.abs(p - T[:3, 3])) < 1e-6)
    return q1, q2, q3, alcanzable


# ==========================================
# EJECUCIÓN Y PRUEBA DE ESCRITORIO
# ==========================================
if __name__ == "__main__":
    qd = np.radians([q1_deg, q2_deg, q3_deg])
    Tdes = fk_T(*qd, L1, L2, L3, L4, L5)

    q1, q2, q3, alcanzable = cinematica_inversa_pata_des(Tdes, L1, L2, L3, L4, L5, codo="arriba")

    print(f"{'='*50}")
    print(f"q deseados: q1 = {q1_deg}°  q2 = {q2_deg}°  q3 = {q3_deg}°")
    print(f"Tdes:\n{np.round(Tdes, 4)}")
    print(f"{'='*50}")

    if not alcanzable:
        print("No alcanzable (Punto fuera del espacio de trabajo)")
    else:
        q1d, q2d, q3d = np.degrees([q1, q2, q3])
        print(f"q resueltos: q1 = {q1d:.2f}°  q2 = {q2d:.2f}°  q3 = {q3d:.2f}°")
        print(f"Error residual T: {np.max(np.abs(fk_T(q1, q2, q3, L1, L2, L3, L4, L5) - Tdes)):.2e}")
        for aviso in joint_limit_warnings(q1d, q2d, q3d):
            print(f"Advertencia: {aviso}")
