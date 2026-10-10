"""
Cinemática inversa de la pata del robot bípedo.

Implementación geométrica (seno/coseno explícitos, roll-pitch-pitch
con offset DH). Modelo viejo: 5 longitudes (L1..L5), sin articulación
fantasma ni L6. Pata IZQUIERDA.
"""

import numpy as np

# Dimensiones físicas del muñeco (mm)
L1, L2, L3, L4, L5 = 51.06, 78.37, 54.2, 138.5, 160.52

# Límites articulares [°]
q1_min, q1_max = 0, 90
q2_min, q2_max = -90, 90
q3_min, q3_max = -90, 90


def cinematica_inversa_pata_geom(x, y, z, L1, L2, L3, L4, L5, codo="arriba"):
    """
    Resuelve q1, q2, q3 [rad] por el método geométrico (plano XZ para
    la cadera roll, plano del "brazo" para rodilla y cadera pitch),
    usando seno/coseno en vez de atan2 con offset.

    codo="arriba" -> sin(q3) positivo
    codo="abajo"  -> sin(q3) negativo

    Retorna (q1, q2, q3, alcanzable).
    """
    # Paso 1: traslación al hombro (plano XZ)
    xp = x - L2
    zp = z + L1                                    # <- IZQUIERDA: z + L1
    rho = np.sqrt(xp**2 + zp**2)
    if rho < 1e-12:
        return None, None, None, False

    # Paso 2: q1 (cadera roll)
    c1 = xp / rho
    s1 = -zp / rho                                 # <- IZQUIERDA: -zp
    q1 = np.arctan2(s1, c1)

    # Paso 3: posición del "brazo" (plano del muslo) y q3
    x_arm = rho - L3
    y_arm = y

    c3 = (x_arm**2 + y_arm**2 - L4**2 - L5**2) / (2*L4*L5)
    if abs(c3) > 1 + 1e-9:
        return None, None, None, False
    c3 = np.clip(c3, -1.0, 1.0)

    s3 = np.sqrt(1 - c3**2) * (1 if codo == "arriba" else -1)
    q3 = np.arctan2(s3, c3)

    # Paso 4: q2 (cadera pitch)
    k1 = L4 + L5*np.cos(q3)
    k2 = L5*np.sin(q3)
    denom2 = k1**2 + k2**2
    c2 = (k1*x_arm - k2*y_arm) / denom2
    s2 = (k2*x_arm + k1*y_arm) / denom2
    q2 = np.arctan2(s2, c2)

    return q1, q2, q3, True


def joint_limit_warnings(q1_deg, q2_deg, q3_deg, tol=1e-6):
    warnings = []
    for nombre, v, mn, mx in (("q1", q1_deg, q1_min, q1_max),
                              ("q2", q2_deg, q2_min, q2_max),
                              ("q3", q3_deg, q3_min, q3_max)):
        if not (mn - tol) <= v <= (mx + tol):
            warnings.append(f"{nombre} fuera de rango [{mn}°, {mx}°]")
    return warnings


if __name__ == "__main__":
    x, y, z = L2 + L3 + L4 + L5, 0, -L1   # home: objetivo con q1=q2=q3=0 (pierna estirada)

    q1, q2, q3, alcanzable = cinematica_inversa_pata_geom(x, y, z, L1, L2, L3, L4, L5, codo="arriba")

    print(f"{'='*50}")
    print(f"Objetivo: x = {x}  y = {y}  z = {z}")
    print(f"{'='*50}")

    if not alcanzable:
        print("No alcanzable")
    else:
        q1d, q2d, q3d = np.degrees([q1, q2, q3])
        print(f"q1 = {q1d:.2f}°  q2 = {q2d:.2f}°  q3 = {q3d:.2f}°")
        for nombre, v, mn, mx in (("q1", q1d, q1_min, q1_max),
                                  ("q2", q2d, q2_min, q2_max),
                                  ("q3", q3d, q3_min, q3_max)):
            if not mn <= v <= mx:
                print(f"Advertencia: {nombre} fuera de rango [{mn}°, {mx}°]")
