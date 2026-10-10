"""
Cinemática inversa
Implementación algebraica (roll-pitch-pitch con offset DH).
Modelo viejo: 5 longitudes (L1..L5), sin articulación fantasma ni L6.
Pierna IZQUIERDA.
"""

import numpy as np

L1, L2, L3, L4, L5 = 51.06,78.37,54.2,138.5,160.52

# Límites articulares [°]
q1_min, q1_max = 0, 90
q2_min, q2_max = -90, 90
q3_min, q3_max = -90, 90


def cinematica_inversa_pata_alg(x, y, z, L1, L2, L3, L4, L5, codo="arriba"):
    xp = x - L2
    zp = z + L1                                    # <- IZQUIERDA: z + L1
    rho = np.sqrt(xp**2 + zp**2)

    q1 = np.arctan2(-zp, xp)                       # <- IZQUIERDA: -zp

    c3 = ((rho - L3)**2 + y**2 - L4**2 - L5**2) / (2*L4*L5)
    if abs(c3) > 1 + 1e-9:                 # tolerancia numérica en el borde
        return None, None, None, False
    c3 = np.clip(c3, -1.0, 1.0)

    s3 = np.sqrt(1 - c3**2) * (1 if codo == "arriba" else -1)
    q3 = np.arctan2(s3, c3)

    A = L4 + L5*np.cos(q3)
    B = L5*np.sin(q3)
    det = A**2 + B**2
    c2 = (A*(rho - L3) - B*y) / det
    s2 = (B*(rho - L3) + A*y) / det
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

    q1, q2, q3, alcanzable = cinematica_inversa_pata_alg(x, y, z, L1, L2, L3, L4, L5, codo="arriba")

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
