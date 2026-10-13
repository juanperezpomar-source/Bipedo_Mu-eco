"""
Generación de trayectorias: perfiles temporales y planificación.

Implementa los métodos de la clase "Manipulator Inverse kinematics II"
(control cinemático y generación de trayectorias):

  1. Lineal                      q(t) = a1 t + a0
  2. Cúbico                      posición y velocidad inicial/final
  3. Quíntico (5º grado)         posición, velocidad y aceleración
  4. Puntos intermedios          cúbico con velocidades de paso

Cada perfil es un polinomio por tramos (clase Perfil) que se evalúa en
posición, velocidad y aceleración. Las unidades son las que se le den
(grados y segundos para articulaciones, mm y segundos para caminos).

Además:

  - planificar_ptp():      movimiento punto a punto de las 3
                           articulaciones, coordinado (isócrono) o
                           independiente.
  - ley_temporal_camino(): ley temporal s(t) sobre un camino cartesiano
                           p(s) (el lápiz recorre el trazo con el
                           perfil elegido).
  - planificar_dibujo():   convierte una trayectoria del trazador en
                           ángulos de la interfaz muestreados a 50 Hz:
                           ley temporal cartesiana en los trazos y
                           movimientos articulares punto a punto con el
                           lápiz levantado.
"""

import math

import numpy as np


# Velocidad máxima del servo, solo para avisos de la interfaz. Valor de la
# pata del parcial (RDS51150 a 12 V: 0.21 s / 60° sin carga). En el muñeco
# (DS3245 / DS3240) el límite que manda es la rampa del firmware
# (VEL_DEG_S en firmware/muneco_motores/src/main.cpp), mucho más lenta.
VEL_MAX_SERVO_DEG_S = 60.0 / 0.21

METODOS = (
    'lineal', 'cubico', 'quintico',
    'puntos_cubico',
)

NOMBRES = {
    'lineal': 'Lineal',
    'cubico': 'Cúbico',
    'quintico': 'Quíntico (5º grado)',
    'puntos_cubico': 'Puntos intermedios - cúbico',
}


class ErrorPerfil(ValueError):
    """Parámetros que no permiten construir el perfil."""


# ----------------------------------------------------------------------
# Polinomio por tramos
# ----------------------------------------------------------------------

class Perfil:
    """Trayectoria 1D por tramos polinómicos.

    tramos: lista de (t_ini, t_fin, coef) con coef en potencias
    crecientes de tau = t - t_ini. Antes de t0 vale q(t0) y después de
    tf vale q(tf), con velocidad y aceleración nulas."""

    def __init__(self, tramos, nombre='', info=None):
        if not tramos:
            raise ErrorPerfil('perfil vacío')
        self.tramos = [(float(a), float(b), np.asarray(c, dtype=float)) for a, b, c in tramos]
        self.nombre = nombre
        self.info = dict(info or {})

    @property
    def t0(self):
        return self.tramos[0][0]

    @property
    def tf(self):
        return self.tramos[-1][1]

    @property
    def duracion(self):
        return self.tf - self.t0

    def evaluar(self, t):
        """(q, q', q'') en los instantes t (escalar o array)."""
        t = np.asarray(t, dtype=float)
        q = np.empty_like(t)
        v = np.zeros_like(t)
        a = np.zeros_like(t)
        inicios = np.array([tr[0] for tr in self.tramos])
        idx = np.clip(np.searchsorted(inicios, t, side='right') - 1, 0, len(self.tramos) - 1)
        for k, (ti, tf_, c) in enumerate(self.tramos):
            m = idx == k
            if not np.any(m):
                continue
            tau = np.clip(t[m], ti, tf_) - ti
            p = np.polynomial.polynomial
            q[m] = p.polyval(tau, c)
            dc = p.polyder(c) if len(c) > 1 else np.array([0.0])
            ddc = p.polyder(dc) if len(dc) > 1 else np.array([0.0])
            v[m] = p.polyval(tau, dc)
            a[m] = p.polyval(tau, ddc)
        fuera = (t < self.t0) | (t > self.tf)
        v[fuera] = 0.0
        a[fuera] = 0.0
        return q, v, a

    def coeficientes_globales(self):
        """Coeficientes de cada tramo en función del tiempo absoluto t
        (como en la presentación), en potencias crecientes."""
        P = np.polynomial.Polynomial
        out = []
        for ti, tf_, c in self.tramos:
            out.append((ti, tf_, P(c)(P([-ti, 1.0])).coef))
        return out

    def texto(self, decimales=4):
        """Ecuaciones q(t) de cada tramo, legibles."""
        lineas = []
        for ti, tf_, c in self.coeficientes_globales():
            terminos = []
            for k in range(len(c) - 1, -1, -1):
                v = round(float(c[k]), decimales)
                if abs(v) < 10 ** -decimales:
                    continue
                pot = '' if k == 0 else ('t' if k == 1 else f't^{k}')
                terminos.append(f"{v:+g}{pot}")
            expr = ' '.join(terminos).lstrip('+') or '0'
            lineas.append(f"q(t) = {expr}   para {ti:g} ≤ t ≤ {tf_:g}")
        return '\n'.join(lineas)


def _poly_local(cond, T):
    """Polinomio en tau que cumple condiciones en tau=0 y tau=T.
    cond = [(orden_derivada, tau, valor), ...]."""
    n = len(cond)
    A = np.zeros((n, n))
    b = np.zeros(n)
    for i, (d, tau, val) in enumerate(cond):
        for k in range(d, n):
            A[i, k] = math.factorial(k) / math.factorial(k - d) * tau ** (k - d)
        b[i] = val
    return np.linalg.solve(A, b)


# ----------------------------------------------------------------------
# 1-5. Punto a punto
# ----------------------------------------------------------------------

def _validar_tiempos(t0, tf):
    if not tf > t0:
        raise ErrorPerfil(f'el tiempo final ({tf:g}) debe ser mayor que el inicial ({t0:g})')


def lineal(q0, qf, t0, tf):
    _validar_tiempos(t0, tf)
    T = tf - t0
    return Perfil([(t0, tf, [q0, (qf - q0) / T])], 'lineal')


def cubico(q0, qf, t0, tf, v0=0.0, vf=0.0):
    _validar_tiempos(t0, tf)
    T = tf - t0
    c = _poly_local([(0, 0, q0), (0, T, qf), (1, 0, v0), (1, T, vf)], T)
    return Perfil([(t0, tf, c)], 'cubico')


def quintico(q0, qf, t0, tf, v0=0.0, vf=0.0, a0=0.0, af=0.0):
    _validar_tiempos(t0, tf)
    T = tf - t0
    c = _poly_local([(0, 0, q0), (0, T, qf), (1, 0, v0), (1, T, vf), (2, 0, a0), (2, T, af)], T)
    return Perfil([(t0, tf, c)], 'quintico')


# ----------------------------------------------------------------------
# 6. Puntos intermedios
# ----------------------------------------------------------------------

def velocidades_de_paso(qs, ts, v0=0.0, vf=0.0):
    """Velocidades en los puntos intermedios (criterio de la clase): cero
    si las pendientes de los tramos vecinos cambian de signo, y si no, la
    media de las dos pendientes."""
    qs, ts = np.asarray(qs, float), np.asarray(ts, float)
    m = np.diff(qs) / np.diff(ts)
    vs = [v0]
    for k in range(1, len(qs) - 1):
        a, b = m[k - 1], m[k]
        vs.append(0.0 if a * b <= 0 else (a + b) / 2)
    vs.append(vf)
    return vs


def cubico_puntos(qs, ts, vs=None, v0=0.0, vf=0.0):
    """Interpolación cúbica por tramos pasando por todos los puntos, con
    las velocidades de paso dadas (o calculadas con velocidades_de_paso)."""
    qs, ts = list(map(float, qs)), list(map(float, ts))
    if len(qs) != len(ts) or len(qs) < 2:
        raise ErrorPerfil('hacen falta al menos 2 puntos, con un tiempo por punto')
    if any(b <= a for a, b in zip(ts, ts[1:])):
        raise ErrorPerfil('los tiempos de paso deben ser crecientes')
    if vs is None:
        vs = velocidades_de_paso(qs, ts, v0, vf)
    tramos = []
    for k in range(len(qs) - 1):
        T = ts[k + 1] - ts[k]
        c = _poly_local([(0, 0, qs[k]), (0, T, qs[k + 1]), (1, 0, vs[k]), (1, T, vs[k + 1])], T)
        tramos.append((ts[k], ts[k + 1], c))
    return Perfil(tramos, 'puntos_cubico', {'velocidades': list(map(float, vs))})


# ----------------------------------------------------------------------
# Construcción a partir de un método y sus parámetros
# ----------------------------------------------------------------------

def construir(metodo, q0, qf, t0=0.0, T=None, params=None):
    """Perfil punto a punto de una articulación.

    params (según el método): v0, vf, a0, af."""
    p = dict(params or {})
    tf = None if T is None else t0 + T
    if metodo == 'lineal':
        return lineal(q0, qf, t0, tf)
    if metodo == 'cubico':
        return cubico(q0, qf, t0, tf, p.get('v0', 0.0), p.get('vf', 0.0))
    if metodo == 'quintico':
        return quintico(q0, qf, t0, tf, p.get('v0', 0.0), p.get('vf', 0.0), p.get('a0', 0.0), p.get('af', 0.0))
    raise ErrorPerfil(f'método desconocido: {metodo}')


def factor_pico(metodo):
    """Velocidad pico / velocidad media para un movimiento en reposo-reposo."""
    return {'lineal': 1.0, 'cubico': 1.5, 'quintico': 1.875, 'puntos_cubico': 1.5}.get(metodo, 1.5)


# ----------------------------------------------------------------------
# Movimiento articular de las 3 articulaciones
# ----------------------------------------------------------------------

def planificar_ptp(q_ini, q_fin, metodo, T=None, params=None, coordinado=True, t0=0.0):
    """Perfiles de las articulaciones para ir de q_ini a q_fin.

    coordinado=True (isócrono): todas empiezan y terminan a la vez, con
    el mismo T."""
    params = dict(params or {})
    if T is None:
        raise ErrorPerfil('falta el tiempo del movimiento')
    out = []
    for a, b in zip(q_ini, q_fin):
        pj = dict(params)
        out.append(construir(metodo, a, b, t0, T, pj))
    return out


def planificar_puntos(qs, ts, metodo, params=None):
    """Perfiles de las 3 articulaciones por puntos intermedios.
    qs: lista de configuraciones (N x 3) [°]; ts: tiempos de paso."""
    params = dict(params or {})
    qs = np.asarray(qs, float)
    if metodo == 'puntos_cubico':
        return [cubico_puntos(qs[:, j], ts, v0=params.get('v0', 0.0), vf=params.get('vf', 0.0))
                for j in range(qs.shape[1])]
    raise ErrorPerfil(f'método de puntos intermedios desconocido: {metodo}')


def muestrear(perfiles, dt=0.02, t0=None, tf=None):
    """Muestrea varios perfiles en una rejilla común. Devuelve
    (t, Q, V, A) con Q, V, A de forma (len(t), n_perfiles)."""
    t0 = min(p.t0 for p in perfiles) if t0 is None else t0
    tf = max(p.tf for p in perfiles) if tf is None else tf
    n = max(2, int(math.ceil((tf - t0) / dt)) + 1)
    t = np.linspace(t0, tf, n)
    Q, V, A = (np.stack(c, -1) for c in zip(*[p.evaluar(t) for p in perfiles]))
    return t, Q, V, A


# ----------------------------------------------------------------------
# Ley temporal sobre un camino cartesiano
# ----------------------------------------------------------------------

def perfil_camino(L, metodo, vmax, amax=None):
    """Ley temporal s(t) para recorrer una longitud L (reposo-reposo) sin
    superar la velocidad vmax ni la aceleración amax (en el lineal la
    aceleración es infinita en los extremos y amax no se puede respetar)."""
    if L <= 1e-9:
        return Perfil([(0.0, 1e-6, [0.0])], metodo)
    if vmax <= 0:
        raise ErrorPerfil('la velocidad del lápiz debe ser positiva')
    T = factor_pico(metodo) * L / vmax
    # Aceleración pico de un movimiento reposo-reposo de longitud L en T:
    # cúbico 6·L/T², quíntico 10/sqrt(3)·L/T². Si se da amax, se alarga T
    # para no superarla (en tramos cortos es la que manda).
    if amax and amax > 0 and metodo in ('cubico', 'quintico'):
        k = 6.0 if metodo == 'cubico' else 10.0 / math.sqrt(3.0)
        T = max(T, math.sqrt(k * L / amax))
    if metodo == 'lineal':
        return lineal(0.0, L, 0.0, T)
    if metodo == 'cubico':
        return cubico(0.0, L, 0.0, T)
    if metodo == 'quintico':
        return quintico(0.0, L, 0.0, T)
    raise ErrorPerfil(f'método no válido para la ley temporal del camino: {metodo}')


# Qué fija cada ley temporal del lápiz. En cada trazo (o lado entre
# esquinas) el lápiz parte QUIETO y llega QUIETO: las condiciones extra de
# los polinomios de mayor grado (velocidad y aceleración en los extremos)
# valen 0 y no se piden. El usuario solo da v (velocidad máxima) y a
# (aceleración máxima); con ellas se calcula la duración T de cada trazo.
CONDICIONES_CAMINO = {
    'lineal': ('1er grado, 2 condiciones: s(0) = 0, s(T) = L. Velocidad constante = v; '
               'arranca y frena de golpe (no usa a).', ('v',)),
    'cubico': ('3er grado, 4 condiciones: s(0) = 0, s(T) = L, ṡ(0) = ṡ(T) = 0. '
               'T = 1.5·L/v, o más si la aceleración superaría a.', ('v', 'a')),
    'quintico': ('5º grado, 6 condiciones: s(0) = 0, s(T) = L, ṡ(0) = ṡ(T) = 0, '
                 's̈(0) = s̈(T) = 0. T = 1.875·L/v, o más si la aceleración superaría a.', ('v', 'a')),
}


def ley_temporal_camino(puntos, metodo, vmax, amax=None, dt=0.02):
    """Recorre la polilínea 'puntos' (N x 3) con la ley s(t).
    Devuelve (t, P) con P (len(t) x 3) muestreado cada dt."""
    P = np.asarray(puntos, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s_nodos = np.concatenate([[0.0], np.cumsum(seg)])
    perfil = perfil_camino(float(s_nodos[-1]), metodo, vmax, amax)
    n = max(2, int(math.ceil(perfil.duracion / dt)) + 1)
    t = np.linspace(0.0, perfil.duracion, n)
    s, _, _ = perfil.evaluar(t)
    s = np.clip(s, 0.0, s_nodos[-1])
    out = np.stack([np.interp(s, s_nodos, P[:, k]) for k in range(3)], -1)
    return t, out


# ----------------------------------------------------------------------
# Planificación de un dibujo completo
#
# Los puntos van en el sistema de las pestañas de IK del equipo, donde +X
# apunta hacia abajo: "arriba" es -X. Se puede pasar otra dirección con el
# argumento arriba (vector unitario).
# ----------------------------------------------------------------------

ARRIBA = (-1.0, 0.0, 0.0)

class Plan:
    """Movimiento listo para ejecutar: ángulos de la interfaz [°]
    muestreados cada dt, con la posición del pie y el estado del lápiz."""

    def __init__(self, t, angulos, posiciones, lapiz_abajo, tramos, dt):
        self.t = np.asarray(t, float)
        self.angulos = np.asarray(angulos, float)
        self.posiciones = np.asarray(posiciones, float)
        self.lapiz_abajo = np.asarray(lapiz_abajo, bool)
        self.tramos = tramos            # [(t_ini, t_fin, descripción)]
        self.dt = dt

    @property
    def duracion(self):
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0

    def derivadas(self):
        """Velocidad y aceleración articulares [°/s, °/s²] (diferencias)."""
        V = np.gradient(self.angulos, self.t, axis=0)
        A = np.gradient(V, self.t, axis=0)
        return V, A

    def en(self, t):
        """Ángulos en el instante t (interpolación lineal entre muestras)."""
        return np.array([np.interp(t, self.t, self.angulos[:, j]) for j in range(3)])


def _ik_camino(P, ik, side, semilla):
    """Cinemática inversa de cada muestra, encadenando la semilla."""
    out = np.empty_like(P)
    for i, p in enumerate(P):
        t, ok, motivo = ik(*p, side=side, semilla_deg=semilla)
        if not ok:
            raise ErrorPerfil(f'punto ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) {motivo}')
        out[i] = t
        semilla = t
    return out


def ptp_muestreado(q_a, q_b, metodo, v_art, params, coordinado, dt):
    """Movimiento articular punto a punto con duración automática según
    la velocidad articular máxima v_art [°/s]."""
    delta = np.max(np.abs(np.asarray(q_b) - np.asarray(q_a)))
    metodo_ptp = metodo if metodo in ('lineal', 'cubico', 'quintico') else 'cubico'
    if delta < 1e-6:
        return np.array([0.0]), np.array([q_a], float)
    p = dict(params)
    T = max(0.2, factor_pico(metodo_ptp) * delta / v_art)
    perfiles = planificar_ptp(q_a, q_b, metodo_ptp, T, p, coordinado)
    t, Q, _, _ = muestrear(perfiles, dt)
    return t - t[0], Q


def dividir_en_esquinas(P, umbral_deg=30.0, ventana_mm=3.0):
    """Índices de las esquinas de la polilínea (incluye el primero y el
    último). El giro en cada vértice se mide entre cuerdas de ventana_mm
    hacia atrás y hacia delante, no entre segmentos consecutivos: así la
    "escalera" de una diagonal o una curva con coordenadas redondeadas a
    milímetros no cuenta como esquina, y una esquina real sí. Se toma el
    vértice de mayor giro de cada zona que supera umbral_deg."""
    P = np.asarray(P, float)
    n = len(P)
    if n < 3:
        return [0, n - 1]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    giro = np.zeros(n)
    for k in range(1, n - 1):
        # Cerca de los extremos la ventana queda recortada y el giro medido
        # no es fiable: ahí no se buscan esquinas.
        if s[k] < ventana_mm or s[-1] - s[k] < ventana_mm:
            continue
        i = np.searchsorted(s, s[k] - ventana_mm, side='right') - 1
        j = np.searchsorted(s, s[k] + ventana_mm, side='left')
        i, j = max(i, 0), min(j, n - 1)
        a, b = P[k] - P[i], P[j] - P[k]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        giro[k] = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b) / (na * nb))))))
    cortes = [0]
    k = 1
    while k < n - 1:
        if giro[k] > umbral_deg:
            # zona contigua por encima del umbral: su vértice de mayor giro
            j = k
            while j + 1 < n - 1 and giro[j + 1] > umbral_deg:
                j += 1
            m = k + int(np.argmax(giro[k:j + 1]))
            if s[m] - s[cortes[-1]] > 1e-6:
                cortes.append(m)
            k = j + 1
        else:
            k += 1
    if cortes[-1] != n - 1:
        cortes.append(n - 1)
    return cortes


def suavizar_lado(P, ventana_mm=3.0, paso_mm=0.5):
    """Suaviza una polilínea SIN esquinas (un lado entre esquinas) con
    una media móvil de ventana_mm, dejando fijos sus extremos. Quita la
    "escalera" que deja redondear las coordenadas a milímetros (desvío
    del orden de medio milímetro), que de otro modo obliga al lápiz a
    cambiar de dirección bruscamente en cada milímetro."""
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    if L < 2 * ventana_mm or len(P) < 3:
        return P
    n = int(math.ceil(L / paso_mm)) + 1
    sr = np.linspace(0.0, L, n)
    R = np.stack([np.interp(sr, s, P[:, k]) for k in range(3)], -1)
    m = max(1, int(round(ventana_mm / paso_mm / 2)))
    out = R.copy()
    for i in range(1, n - 1):
        h = min(m, i, n - 1 - i)          # ventana simétrica que no se sale
        out[i] = R[i - h:i + h + 1].mean(axis=0)
    out[0], out[-1] = P[0], P[-1]
    return out


def planificar_dibujo(puntos, side, t_actual_deg, ik, fk_pie,
                      metodo_camino='quintico', v_lapiz=30.0, a_lapiz=100.0,
                      metodo_articular='quintico', v_articular=30.0, params_articular=None,
                      coordinado=True, dt=0.02, margen_lapiz=0.5, margen_seguridad=3.0,
                      parar_en_esquinas=True, umbral_esquina_deg=30.0, suavizar=True,
                      arriba=ARRIBA):
    """Convierte una trayectoria (N x 3 puntos en mm, sistema del equipo)
    en un Plan ejecutable.

    - Lápiz abajo (el papel es la altura más baja del archivo; la altura
      se mide a lo largo de 'arriba'): ley temporal
      cartesiana 'metodo_camino' en cada trazo, con velocidad v_lapiz
      [mm/s] (y aceleración a_lapiz [mm/s²] si el trazo es corto y la
      curva la superaría), y cinemática inversa en cada muestra. Con
      parar_en_esquinas, cada
      tramo entre esquinas (giro > umbral_esquina_deg) tiene su propia
      ley temporal y el lápiz se detiene en la esquina: así la
      aceleración no salta al cambiar de dirección. Con suavizar, cada
      lado se suaviza (suavizar_lado) para quitar los escalones del
      redondeo a milímetros.
    - Lápiz arriba: subir en vertical (ley cartesiana), desplazarse con un
      movimiento ARTICULAR punto a punto ('metodo_articular', velocidad
      v_articular [°/s]) y bajar en vertical. Si el movimiento articular
      bajara a menos de margen_seguridad mm del papel, ese desplazamiento
      se hace en línea recta cartesiana.
    - Antes de empezar: movimiento articular desde la pose actual hasta
      encima del primer punto.

    ik(x, y, z, side, semilla_deg) -> (t_deg, ok, motivo)
    fk_pie(t_deg, side) -> posición del pie [mm]"""
    P = np.asarray(puntos, float)
    if len(P) < 2:
        raise ErrorPerfil('la trayectoria necesita al menos 2 puntos')
    params_articular = dict(params_articular or {})
    up = np.asarray(arriba, float)
    altura = P @ up
    h_papel = float(altura.min())
    abajo = altura <= h_papel + margen_lapiz
    lift = float(altura.max() - h_papel) or 20.0

    ts, angs, poss, lap, tramos = [], [], [], [], []
    t_acum = [0.0]

    def agregar(t_local, Q, Pp, pen, desc):
        if len(t_local) == 0:
            return
        base = t_acum[0]
        if ts:          # evita duplicar el instante de unión
            t_local, Q, Pp, pen = t_local[1:], Q[1:], Pp[1:], pen[1:]
            if len(t_local) == 0:
                return
        ts.extend(base + t_local)
        angs.extend(Q)
        poss.extend(Pp)
        lap.extend(pen)
        t_ini = base + (t_local[0] if len(t_local) else 0.0)
        t_acum[0] = base + float(t_local[-1])
        tramos.append((t_ini, t_acum[0], desc))

    def tramo_cartesiano(puntos_tramo, pen, desc):
        tt, Pc = ley_temporal_camino(puntos_tramo, metodo_camino, v_lapiz, a_lapiz, dt)
        Q = _ik_camino(Pc, ik, side, angs[-1] if angs else t_actual_deg)
        agregar(tt, Q, Pc, np.full(len(tt), pen), desc)

    def tramo_articular(p_destino, desc):
        q_a = np.asarray(angs[-1] if angs else t_actual_deg, float)
        q_b, ok, motivo = ik(*p_destino, side=side, semilla_deg=tuple(q_a))
        if not ok:
            raise ErrorPerfil(f'punto ({p_destino[0]:.1f}, {p_destino[1]:.1f}, {p_destino[2]:.1f}) {motivo}')
        tt, Q = ptp_muestreado(q_a, q_b, metodo_articular, v_articular, params_articular, coordinado, dt)
        Pp = np.array([fk_pie(q, side) for q in Q])
        # ¿el pie baja demasiado cerca del papel durante el movimiento articular?
        h = Pp @ up
        if angs and h.min() < h_papel + margen_seguridad and h[0] > h_papel + margen_lapiz:
            tramo_cartesiano(np.array([Pp[0], p_destino]), False, desc + ' (en línea recta: el articular rozaba el papel)')
            return
        agregar(tt, Q, Pp, np.zeros(len(tt), bool), desc)

    # Separar en tramos de lápiz abajo / arriba
    runs = []
    i = 0
    while i < len(P):
        j = i
        while j + 1 < len(P) and abajo[j + 1] == abajo[i]:
            j += 1
        runs.append((bool(abajo[i]), i, j))
        i = j + 1

    # Aproximación: pose actual -> encima del primer punto -> bajar
    primero = P[0]
    encima = primero + up * lift
    tramo_articular(encima, 'aproximación (articular)')
    if abajo[0]:
        tramo_cartesiano(np.array([encima, primero]), False, 'bajar el lápiz')

    n_trazo = 0
    for k, (es_abajo, i, j) in enumerate(runs):
        if es_abajo:
            n_trazo += 1
            if j > i:
                trazo = P[i:j + 1]
                cortes = dividir_en_esquinas(trazo, umbral_esquina_deg) if parar_en_esquinas else [0, len(trazo) - 1]
                for a_, b_ in zip(cortes, cortes[1:]):
                    sufijo = f', lado {cortes.index(b_)}/{len(cortes) - 1}' if len(cortes) > 2 else ''
                    lado = trazo[a_:b_ + 1]
                    if suavizar:
                        lado = suavizar_lado(lado)
                    tramo_cartesiano(lado, True,
                                     f'trazo {n_trazo}{sufijo} (ley temporal cartesiana)')
            continue
        # Lápiz arriba entre dos trazos: subir, desplazarse, bajar
        a = P[i - 1] if i > 0 else P[i]
        b = P[j + 1] if j + 1 < len(P) else P[j]
        a_up = a + up * lift
        b_up = b + up * lift
        if i > 0:
            tramo_cartesiano(np.array([a, a_up]), False, 'subir el lápiz')
        tramo_articular(b_up, 'desplazamiento con el lápiz arriba (articular)')
        if j + 1 < len(P):
            tramo_cartesiano(np.array([b_up, b]), False, 'bajar el lápiz')

    # Al terminar, levantar el lápiz
    if abajo[-1]:
        tramo_cartesiano(np.array([P[-1], P[-1] + up * lift]), False, 'levantar el lápiz al terminar')

    return Plan(ts, angs, poss, lap, tramos, dt)


def planificar_por_puntos(puntos, side, t_actual_deg, ik, fk_pie, metodo, v_lapiz=30.0,
                          params=None, dt=0.02, paso_mm=5.0, arriba=ARRIBA):
    """Alternativa: interpolación ARTICULAR por puntos intermedios a lo
    largo de toda la trayectoria (cúbico con velocidades de paso o lineal
    con ajuste parabólico). Los puntos de paso se toman cada paso_mm mm y
    sus tiempos se reparten según la distancia a velocidad v_lapiz. A
    diferencia de la ley cartesiana, entre puntos de paso el pie no sigue
    exactamente la recta del dibujo."""
    params = dict(params or {})
    P = np.asarray(puntos, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    idx = [0]
    for k in range(1, len(P)):
        if s[k] - s[idx[-1]] >= paso_mm:
            idx.append(k)
    # El último punto siempre es de paso. Si cae muy cerca del anterior,
    # lo reemplaza: un tramo de casi 0 mm duraría casi 0 s y ninguna
    # aceleración alcanzaría para la mezcla parabólica.
    if idx[-1] != len(P) - 1:
        if len(idx) > 1 and s[-1] - s[idx[-1]] < paso_mm / 2:
            idx[-1] = len(P) - 1
        else:
            idx.append(len(P) - 1)
    Pv = P[idx]
    Q = _ik_camino(Pv, ik, side, t_actual_deg)
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Pv, axis=0), axis=1))])
    tiempos = dist / max(v_lapiz, 1e-6)
    tiempos = np.maximum.accumulate(tiempos + np.arange(len(tiempos)) * 1e-3)
    perfiles = planificar_puntos(Q, tiempos, metodo, params)
    t, Qs, _, _ = muestrear(perfiles, dt)
    # Aproximación articular desde la pose actual hasta el primer punto
    ta, Qa = ptp_muestreado(np.asarray(t_actual_deg, float), Q[0], 'quintico',
                             params.get('v_articular', 30.0), {}, True, dt)
    t_total = np.concatenate([ta, ta[-1] + t[1:] - t[0]]) if len(ta) > 1 else t - t[0]
    Q_total = np.concatenate([Qa, Qs[1:]]) if len(ta) > 1 else Qs
    Pp = np.array([fk_pie(q, side) for q in Q_total])
    up = np.asarray(arriba, float)
    h_papel = float((P @ up).min())
    tramos = []
    if len(ta) > 1:
        tramos.append((0.0, float(ta[-1]), 'aproximación (articular)'))
    tramos.append((float(t_total[len(ta) - 1]), float(t_total[-1]),
                   f'{NOMBRES[metodo]} ({len(Pv)} puntos de paso)'))
    return Plan(t_total, Q_total, Pp, Pp @ up <= h_papel + 0.5, tramos, dt)
