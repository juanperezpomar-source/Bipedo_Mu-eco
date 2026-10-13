"""
Cinemática de la pierna del MUÑECO bípedo para la generación de
trayectorias (pestaña Trayectorias de la interfaz, trazador y análisis del
espacio de trabajo). Es el equivalente de kinem_v6.py de la pata del
parcial, pero con el modelo del muñeco: cadena 0A4, cinco longitudes
(L1..L5), sin articulación fantasma ni L6.

SISTEMA DE COORDENADAS (el mismo de las pestañas de cinemática inversa,
cinematica_directa_der_MUNECO_izq.py, en mm):
    +X a lo largo de la pierna, hacia ABAJO (vertical).
    +Y hacia adelante.
    Z lateral: la pierna izquierda está en -Z y la derecha en +Z.
    Un papel horizontal es un plano X = constante; "subir" el lápiz es
    restar a X.
    Pie con la pierna colgando:  izq (431.59, 0, -51.06)
                                 der (431.59, 0,  51.06)
test/test_kinem_muneco.py comprueba que posicion_pie() coincide con
cinematica_directa_der_MUNECO_izq.

ÁNGULOS
    En el muñeco el ángulo de la interfaz ES el q de la tabla DH:
    t = (roll, pitch, rodilla) [°] = (q1, q2, q3), t = 0 = pierna
    colgando. Las funciones interfaz_a_modelo / modelo_a_interfaz solo
    convierten grados <-> radianes (se mantienen para tener la misma API
    que kinem_v6).

MODELO (pierna izquierda):

        theta   d     a     alpha
  0A1    0     -L1    L2    -90°
  1A2    q1     0     L3     90°   <- cadera roll
  2A3    q2     0     L4    180°   <- cadera pitch
  3A4    q3     0     L5      0    <- rodilla

    La pierna derecha es la misma cadena con d1 = +L1 y los alpha de
    0A1/1A2 cambiados de signo; es el espejo en Z de la izquierda:
        T_der(q) = espejo_z(T_izq(q))     (mismos q, z -> -z)

Los métodos de cinemática inversa devuelven TODAS las soluciones (las dos
raíces de la cadera roll y los dos codos) y se elige la que respeta los
límites del control. Sin esto, control_node recortaría el ángulo en
silencio y el pie no llegaría al punto pedido.
"""

import math

import numpy as np


# ----------------------------------------------------------------------
# Parámetros geométricos [mm] (cinematica_directa_der_MUNECO_izq.py)
# ----------------------------------------------------------------------

L1, L2, L3, L4, L5 = 51.06, 78.37, 54.2, 138.5, 160.52


# ----------------------------------------------------------------------
# Límites que aplica la interfaz [°] (robot_teleop/teleop_node.py,
# lower_deg / upper_deg). Mismos que los q*_min/q*_max de los módulos
# kinem_invers_leg_*_MUNECO_izq.py.
# ----------------------------------------------------------------------

LIMITES_CONTROL_DEG = ((0.0, 90.0), (-90.0, 90.0), (-90.0, 90.0))


# ----------------------------------------------------------------------
# Paso a RViz. Los puntos de la cinemática están en el marco base de la
# DH; en el URDF del muñeco (Muneco_Bipedo_URDF_V3) el Base_link tiene
# los ejes girados ~-90° en Z respecto a esa base:
#     X_DH (abajo)    -> -Y del Base_link
#     Y_DH (adelante) -> +X del Base_link
#     Z_DH (lateral)  -> +Z del Base_link
# más un desplazamiento del origen. teleop_node.publish_trail usa estos
# valores para el rastro del pie en RViz.
#
# Se obtuvieron ajustando (mínimos cuadrados, 600 posturas al azar) la
# posición del pie de la DH contra el pie del URDF, tomando como efector
# el centro de la plancha del pie: punto (-162.4, 0, -41.1) mm en
# Left_Knee_Link y (-160.8, -3.0, -41.0) mm en Right_Knee_Link (la
# canilla está corrida ~41 mm del eje de la rodilla, sobre ese eje).
# Error resultante: ~1 mm en la zona de dibujo y < 7 mm en todo el rango.
#
# Vale con el Knee Pitch SIN invertir hacia RViz (control_node,
# invert_knee_rviz=False): con +rodilla el URDF y la DH llevan el pie
# hacia atrás.
# ----------------------------------------------------------------------

R_DH_A_BASE_LINK = {
    'left': np.array([[-0.0026, 1.0, -0.0073],
                      [-1.0, -0.0026, 0.0025],
                      [0.0025, 0.0074, 1.0]]),
    'right': np.array([[-0.0156, 0.9999, -0.0051],
                       [-0.9999, -0.0156, -0.0069],
                       [-0.0070, 0.0050, 1.0]]),
}

OFFSET_BASE_LINK_MM = {'left': np.array([-1.2, 28.6, 0.0]),
                       'right': np.array([0.2, 28.9, 0.1])}


# ----------------------------------------------------------------------
# Papel recomendado (resultado del workspace del trazador,
# scripts/trazador/trazador.py, pestaña "Workspace YZ", con los límites
# de arriba, malla de 1 mm): plano X = 364, es decir ~68 mm por encima
# del pie colgando, que es la X con el mayor cuadrado seguro (~160 mm);
# se deja un lado de 150 mm para tener margen en los bordes. En toda el
# área (y con el lápiz levantado 10 mm) queda >= 5° de margen a los
# límites y la rodilla >= 10° de la singularidad.
# ----------------------------------------------------------------------

PAPEL_RECOMENDADO = {
    'left': {'x_papel': 364.0, 'centro': (0.0, -157.0), 'lado': 150.0, 'levantar': 10.0},
    'right': {'x_papel': 364.0, 'centro': (0.0, 157.0), 'lado': 150.0, 'levantar': 10.0},
}


def pose_preparacion(side='left'):
    """Punto encima del centro del papel recomendado, con el lápiz
    levantado (X menor = más arriba). La pierna colgando queda POR DEBAJO
    del papel, así que hay que llevarla aquí antes de colocar la mesa."""
    papel = PAPEL_RECOMENDADO[_lado(side)]
    y, z = papel['centro']
    return (papel['x_papel'] - papel['levantar'], y, z)


def _lado(side):
    side = str(side).lower()
    if side not in ('left', 'right'):
        raise ValueError(f"side debe ser 'left' o 'right', no {side!r}")
    return side


# ----------------------------------------------------------------------
# Conversión de ángulos (en el muñeco t = q, solo cambian las unidades)
# ----------------------------------------------------------------------

def _envolver_deg(a):
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def interfaz_a_modelo(t_deg, side='left'):
    """t = (roll, pitch, rodilla) [°] -> q del modelo DH [rad]."""
    _lado(side)
    return tuple(np.radians(v) for v in t_deg)


def modelo_a_interfaz(q_rad, side='left'):
    """q del modelo DH [rad] -> t = (roll, pitch, rodilla) [°] en (-180, 180]."""
    _lado(side)
    return tuple(_envolver_deg(np.degrees(v)) for v in q_rad)


def margen_limites_deg(t_deg, limites=LIMITES_CONTROL_DEG):
    """Distancia [°] al límite más cercano (negativa si se sale)."""
    return min(m for v, (lo, hi) in zip(t_deg, limites) for m in (v - lo, hi - v))


def avisos_limites(t_deg, limites=LIMITES_CONTROL_DEG):
    nombres = ('roll', 'pitch', 'rodilla')
    return [f"{n} = {v:.1f}° fuera de [{lo:g}°, {hi:g}°]"
            for n, v, (lo, hi) in zip(nombres, t_deg, limites) if not lo <= v <= hi]


# ----------------------------------------------------------------------
# Cinemática directa
# ----------------------------------------------------------------------

def dh_matrix(theta, d, a, alpha):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def fk_modelo(q_rad, side='left'):
    """MTH acumuladas (T01, T02, T03, T04) para q del modelo [rad]."""
    q1, q2, q3 = q_rad
    if _lado(side) == 'left':
        A01 = dh_matrix(0.0, -L1, L2, -math.pi / 2)
        A12 = dh_matrix(q1, 0.0, L3, math.pi / 2)
    else:
        A01 = dh_matrix(0.0, L1, L2, math.pi / 2)
        A12 = dh_matrix(q1, 0.0, L3, -math.pi / 2)
    A23 = dh_matrix(q2, 0.0, L4, math.pi)
    A34 = dh_matrix(q3, 0.0, L5, 0.0)
    T01 = A01
    T02 = T01 @ A12
    T03 = T02 @ A23
    T04 = T03 @ A34
    return T01, T02, T03, T04


# ----------------------------------------------------------------------
# Cambio de sistema: modelo interno <-> sistema del equipo. En el muñeco
# los dos coinciden (la tabla DH ya está escrita en el sistema del
# equipo); se mantienen para tener la misma API que kinem_v6, que usa el
# trazador (vectorizable: p puede ser un array (..., 3)).
# ----------------------------------------------------------------------

def _modelo_a_equipo(p, side='left'):
    _lado(side)
    return np.asarray(p, dtype=float)


def _equipo_a_modelo(p, side='left'):
    _lado(side)
    return np.asarray(p, dtype=float)


def posicion_pie(t_deg, side='left'):
    """Posición del pie (x, y, z) [mm], sistema del equipo, para los
    ángulos de la interfaz t [°]."""
    T = fk_modelo(interfaz_a_modelo(t_deg, side), side)[-1]
    return _modelo_a_equipo(T[:3, 3], side)


def posicion_home(side='left'):
    """Pie con la pierna colgando (todos los ángulos en 0)."""
    return posicion_pie((0.0, 0.0, 0.0), side)


def punto_a_base_link(p, side='left'):
    """Punto del sistema del equipo [mm] -> Base_link del URDF [mm]."""
    side = _lado(side)
    return R_DH_A_BASE_LINK[side] @ np.asarray(p, dtype=float) + OFFSET_BASE_LINK_MM[side]


# ----------------------------------------------------------------------
# Cinemática inversa - pierna izquierda
#
# Cada función devuelve TODAS las soluciones como lista de
# (q1, q2, q3, ok) [rad] del modelo izquierdo. Son vectorizables: x, y, z
# pueden ser arrays (lo usa el análisis del espacio de trabajo); ok es
# la máscara de validez.
# ----------------------------------------------------------------------

def _soluciones_geometrico_izq(x, y, z):
    """Método geométrico (kinem_invers_leg_Geometrico_MUNECO_izq), con
    las dos raíces de la cadera roll (rho > 0 y rho < 0) y los dos codos."""
    xp, zp = x - L2, z + L1
    rho_abs = np.sqrt(xp**2 + zp**2)
    sols = []
    for signo_r in (1.0, -1.0):
        rho = signo_r * rho_abs
        with np.errstate(invalid='ignore', divide='ignore'):
            q1 = np.arctan2(-zp / rho, xp / rho)
        x_arm, y_arm = rho - L3, y
        c3 = (x_arm**2 + y_arm**2 - L4**2 - L5**2) / (2 * L4 * L5)
        ok = (rho_abs > 1e-9) & (np.abs(c3) <= 1 + 1e-9)
        c3 = np.clip(c3, -1.0, 1.0)
        for signo_s3 in (1.0, -1.0):
            q3 = np.arctan2(signo_s3 * np.sqrt(1 - c3**2), c3)
            k1, k2 = L4 + L5 * np.cos(q3), L5 * np.sin(q3)
            den2 = k1**2 + k2**2
            q2 = np.arctan2((k2 * x_arm + k1 * y_arm) / den2, (k1 * x_arm - k2 * y_arm) / den2)
            sols.append((q1, q2, q3, ok))
    return sols


# ----------------------------------------------------------------------
# Adaptación a cualquier pierna (la derecha es el espejo en Z)
# ----------------------------------------------------------------------

def _a_izquierda(p, side):
    """Punto de la pierna 'side' expresado para los métodos izquierdos."""
    x, y, z = p
    return (x, y, z) if side == 'left' else (x, y, -z)


def _q_desde_izquierda(q, side):
    """Ángulos de los métodos izquierdos -> ángulos de la pierna 'side'
    (iguales: el espejo en Z no cambia los ángulos)."""
    return tuple(q)


def _elegir(candidatos, p, side, semilla_deg=None):
    """De las soluciones (q del modelo), devuelve la mejor en ángulos de la
    interfaz: dentro de los límites del control, con la rodilla en
    flexión (t3 >= 0) y, entre esas, la más cercana a la semilla (o la de
    mayor margen a los límites si no hay semilla)."""
    validas, fuera = [], []
    for q in candidatos:
        if not np.all(np.isfinite(q)):
            continue
        if np.linalg.norm(fk_modelo(q, side)[-1][:3, 3] - p) > 1e-6:
            continue
        t = tuple(float(v) for v in modelo_a_interfaz(q, side))
        (validas if margen_limites_deg(t) >= 0 else fuera).append(t)
    if not validas:
        motivo = 'fuera de los límites del control' if fuera else 'fuera del alcance de la pierna'
        return None, False, motivo
    flexion = [t for t in validas if t[2] >= 0] or validas
    if semilla_deg is not None:
        mejor = min(flexion, key=lambda t: max(abs(_envolver_deg(a - b)) for a, b in zip(t, semilla_deg)))
    else:
        mejor = max(flexion, key=margen_limites_deg)
    return mejor, True, ''


def ik_geometrico(x, y, z, side='left', semilla_deg=None):
    """Cinemática inversa por el método geométrico. (x, y, z) en el
    sistema del equipo [mm].
    Retorna (t_deg, alcanzable, motivo): t en el convenio de la interfaz."""
    side = _lado(side)
    p = _equipo_a_modelo((x, y, z), side)
    cand = [_q_desde_izquierda((float(a), float(b), float(c)), side)
            for a, b, c, ok in _soluciones_geometrico_izq(*_a_izquierda(p, side)) if bool(ok)]
    return _elegir(cand, p, side, semilla_deg)


if __name__ == '__main__':
    for lado in ('left', 'right'):
        home = posicion_home(lado)
        print(f"{lado}: pie colgando {np.round(home, 2)} -> Base_link "
              f"{np.round(punto_a_base_link(home, lado), 1)}")
        objetivo = np.array(pose_preparacion(lado))
        t, ok, motivo = ik_geometrico(*objetivo, side=lado)
        print(f"  preparación {objetivo}: {np.round(t, 2) if ok else motivo}")
