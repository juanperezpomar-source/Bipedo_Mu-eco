"""
Cinemática inversa de la pata del robot bípedo.

Implementación por Newton-Raphson (Jacobiano con pseudoinversa amortiguada),
con verificación geométrica previa de alcanzabilidad.
Modelo viejo: 5 longitudes (L1..L5), sin articulación fantasma ni L6.
Pata IZQUIERDA.
"""

import numpy as np

# Dimensiones físicas del muñeco (mm)
L1, L2, L3, L4, L5 = 51.06, 78.37, 54.2, 138.5, 160.52

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


def calcular_jacobiano_rad(q_rad, L1, L2, L3, L4, L5):
    q1, q2, q3 = q_rad
    A01 = dh_matrix(0,  d=-L1, a=L2, alpha=-np.pi/2)
    A12 = dh_matrix(q1, d=0,   a=L3, alpha=np.pi/2)
    A23 = dh_matrix(q2, d=0,   a=L4, alpha=np.pi)
    A34 = dh_matrix(q3, d=0,   a=L5, alpha=0)
    T04 = A01 @ A12 @ A23 @ A34
    p_e = T04[0:3, 3]
    T01 = A01
    T02 = T01 @ A12
    T03 = T02 @ A23
    J1 = np.cross(T01[0:3, 2], (p_e - T01[0:3, 3]))
    J2 = np.cross(T02[0:3, 2], (p_e - T02[0:3, 3]))
    J3 = np.cross(T03[0:3, 2], (p_e - T03[0:3, 3]))
    return np.column_stack((J1, J2, J3)), p_e


def es_geometricamente_alcanzable(x, y, z, L1, L2, L3, L4, L5):
    """
    Verificación RÁPIDA y EXACTA de alcanzabilidad, usando las mismas
    condiciones del método algebraico cerrado -- antes de gastar
    ni una iteración de Newton. No depende de en qué q_actual esté
    parado el algoritmo (a diferencia de det(J), que sí depende de eso).
    """
    xp = x - L2
    zp = z + L1                                   # <- IZQUIERDA: z + L1
    rho = np.sqrt(xp**2 + zp**2)

    x_arm = rho - L3
    y_arm = y
    c3 = (x_arm**2 + y_arm**2 - L4**2 - L5**2) / (2*L4*L5)
    if abs(c3) > 1 + 1e-9:
        return False, "fuera de alcance angular (|c3| > 1, el objetivo está más lejos de lo que L4+L5 permiten)"

    return True, "alcanzable"


def cinematica_inversa_newton(objetivo, q_inicial_grados, tolerancia=0.1, max_iter=1000):
    # --- Paso 0: verificación geométrica previa (independiente del Jacobiano) ---
    alcanzable_geo, motivo = es_geometricamente_alcanzable(
        objetivo[0], objetivo[1], objetivo[2], L1, L2, L3, L4, L5)
    if not alcanzable_geo:
        return None, None, False, 0, motivo

    q_actual = np.radians(q_inicial_grados)
    q_min_rad = np.radians([q1_min, q2_min, q3_min])
    q_max_rad = np.radians([q1_max, q2_max, q3_max])

    max_salto_rad = np.radians(40.0)   # freno de emergencia
    lam = 20.0                          # amortiguación (Levenberg-Marquardt)

    for iteracion in range(max_iter):
        J, p_actual = calcular_jacobiano_rad(q_actual, L1, L2, L3, L4, L5)
        error = objetivo - p_actual
        distancia = np.linalg.norm(error)

        if distancia < tolerancia:
            return np.degrees(q_actual), p_actual, True, iteracion, "convergió"

        JJt = J @ J.T
        salto_sugerido = J.T @ np.linalg.solve(JJt + (lam**2) * np.eye(3), error)

        magnitud_salto = np.linalg.norm(salto_sugerido)
        if magnitud_salto > max_salto_rad:
            salto_sugerido = salto_sugerido * (max_salto_rad / magnitud_salto)

        q_actual = q_actual + salto_sugerido
        q_actual = np.clip(q_actual, q_min_rad, q_max_rad)

    # geométricamente alcanzable, pero el algoritmo no convergió (se trabó)
    return np.degrees(q_actual), p_actual, False, max_iter, "no convergió (objetivo sí es alcanzable, pero el algoritmo se trabó)"


def cinematica_inversa_pata_newton(x, y, z,
                                   q1_seed_deg, q2_seed_deg, q3_seed_deg):
    """
    Misma interfaz que los demás métodos (la usan ik_newton_node y la
    interfaz): punto (x, y, z) [mm] y semilla [°].

    Retorna (q1, q2, q3, alcanzable) con los ángulos en radianes.
    alcanzable es False si no es alcanzable o no converge dentro de
    max_iter.
    """
    q_deg, _, alcanzable, _, _ = cinematica_inversa_newton(
        np.array([x, y, z], dtype=float),
        np.array([q1_seed_deg, q2_seed_deg, q3_seed_deg], dtype=float))

    if not alcanzable:
        return None, None, None, False

    q1, q2, q3 = np.radians(q_deg)
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
    # objetivo de prueba: home (q1=q2=q3=0, pierna estirada)
    for objetivo_deseado in [np.array([L2 + L3 + L4 + L5, 0, -L1])]:
        angulos_iniciales = np.array([0.0, 0.0, 0.0])   # semilla no singular
        resultado = cinematica_inversa_newton(objetivo_deseado, angulos_iniciales)
        angulos_finales, posicion_alcanzada, alcanzable, iteraciones, motivo = resultado

        print(f"{'='*50}")
        print(f"Objetivo cartesiano: {np.round(objetivo_deseado, 4)}")
        print(f"{'='*50}")

        if not alcanzable and angulos_finales is None:
            print(f"NO ALCANZABLE (verificación geométrica): {motivo}")
        elif not alcanzable:
            q1d, q2d, q3d = angulos_finales
            print(f"{motivo} ({iteraciones} iter)")
            print(f"q1 = {q1d:.2f}°  q2 = {q2d:.2f}°  q3 = {q3d:.2f}°")
            print("Posición final:", np.round(posicion_alcanzada, 4))
        else:
            q1d, q2d, q3d = angulos_finales
            print(f"¡Éxito! {motivo} en la iteración {iteraciones}")
            print(f"q1 = {q1d:.2f}°  q2 = {q2d:.2f}°  q3 = {q3d:.2f}°")
            print("Posición final:", np.round(posicion_alcanzada, 4))
            print("Error residual (mm):", np.max(np.abs(posicion_alcanzada - objetivo_deseado)))
            for aviso in joint_limit_warnings(q1d, q2d, q3d):
                print(f"Advertencia: {aviso}")
        print()