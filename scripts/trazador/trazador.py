#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trazador YZ para cinemática inversa de la pierna del MUÑECO.

Es el trazador de la pata del parcial con el modelo del muñeco
(robot_kinematics.kinem_muneco: cadena 0A4, L1..L5). El papel, los
límites del lienzo y la elevación del lápiz por defecto salen de
kinem_muneco.PAPEL_RECOMENDADO.

Lienzo: Z horizontal, Y vertical, ambos con limites absolutos configurables.
Al dibujar X es constante. El lapiz se levanta reduciendo X entre figuras.
Guardar y Exportar producen puntos XYZ en milimetros, compatibles con ROS.
La simulación temporal utiliza, cuando esta instalado, perfiles_temporales
(lineal, cubico, quintico) de robot_kinematics; no necesita hardware.
"""
import math
import os
import re
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

CANVAS_MAX_PX = 600   # tamaño máximo del área de dibujo en pantalla
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
KINEMATICS_PKG = os.path.join(REPO, "ros2_ws", "src", "robot_kinematics")
LEG_TRAJ_DIR = os.path.join(KINEMATICS_PKG, "trayectorias")
PAD = 34              # margen para las etiquetas de los ejes
GRID_MM = 10          # separación de la cuadrícula


# =============================================================================
#  Geometría (independiente de la interfaz)
# =============================================================================

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def line_dist(p, a, b):
    """Distancia de p a la recta que pasa por a y b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    if L == 0:
        return dist(p, a)
    return abs(dx * (a[1] - p[1]) - dy * (a[0] - p[0])) / L


def seg_dist(p, a, b):
    """Distancia de p al segmento ab."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return dist(p, (a[0] + t * dx, a[1] + t * dy))


def path_length(pts):
    return sum(dist(a, b) for a, b in zip(pts, pts[1:]))


def resample(pts, spacing):
    """Remuestrea una polilínea con puntos equiespaciados (el ratón no
    entrega eventos a intervalos regulares)."""
    if len(pts) < 2:
        return list(pts)
    out = [pts[0]]
    acc = 0.0
    prev = pts[0]
    for p in pts[1:]:
        d = dist(prev, p)
        while acc + d >= spacing and d > 0:
            t = (spacing - acc) / d
            q = (prev[0] + t * (p[0] - prev[0]), prev[1] + t * (p[1] - prev[1]))
            out.append(q)
            prev, acc = q, 0.0
            d = dist(prev, p)
        acc += d
        prev = p
    if dist(out[-1], pts[-1]) > 1e-9:
        out.append(pts[-1])
    return out


def rdp(pts, eps):
    """Simplificación Ramer-Douglas-Peucker (iterativa)."""
    if len(pts) < 3:
        return list(pts)
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        dmax, idx = 0.0, -1
        for k in range(i + 1, j):
            d = line_dist(pts[k], pts[i], pts[j])
            if d > dmax:
                dmax, idx = d, k
        if dmax > eps:
            keep[idx] = True
            stack += [(i, idx), (idx, j)]
    return [p for p, k in zip(pts, keep) if k]


def prune_closed(verts, eps):
    """Elimina vértices casi colineales o lados muy cortos de un polígono
    cerrado (p. ej. el 'ganchito' que queda al cerrar la figura a mano)."""
    v = list(verts)
    changed = True
    while changed and len(v) > 3:
        changed = False
        for i in range(len(v)):
            a, b, c = v[i - 1], v[i], v[(i + 1) % len(v)]
            if line_dist(b, a, c) < eps or dist(a, b) < eps:
                del v[i]
                changed = True
                break
    return v


def signed_area(v):
    return 0.5 * sum(v[i - 1][0] * v[i][1] - v[i][0] * v[i - 1][1] for i in range(len(v)))


def polygon_name(v):
    n = len(v)
    names = {3: "triángulo", 5: "pentágono", 6: "hexágono",
             7: "heptágono", 8: "octágono"}
    if n == 4:
        ang_ok = True
        for i in range(4):
            a, b, c = v[i - 1], v[i], v[(i + 1) % 4]
            u = (a[0] - b[0], a[1] - b[1])
            w = (c[0] - b[0], c[1] - b[1])
            cosang = (u[0] * w[0] + u[1] * w[1]) / (math.hypot(*u) * math.hypot(*w) + 1e-12)
            if abs(math.degrees(math.acos(max(-1, min(1, cosang)))) - 90) > 15:
                ang_ok = False
        if ang_ok:
            sides = [dist(v[i - 1], v[i]) for i in range(4)]
            return "cuadrado" if max(sides) / min(sides) < 1.2 else "rectángulo"
        return "cuadrilátero"
    return names.get(n, f"polígono de {n} lados")


def analyze_stroke(raw, mode="figura", sens_pct=6.0, close_auto=True, step=1.0):
    """Analiza un trazo (lista de puntos en mm).
    Devuelve dict(name, verts, closed) o None si el trazo es despreciable."""
    pts = resample(raw, 0.5)
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    if diag < 1.0:
        return None

    closed = (close_auto and len(pts) > 4
              and dist(pts[0], pts[-1]) < max(0.15 * diag, 2.0)
              and path_length(pts) > 1.8 * diag)

    # ---- Trazo libre: conserva la forma, solo quita ruido ----
    if mode == "libre":
        if closed:
            v = rdp(pts + [pts[0]], 0.3)[:-1]
            return {"name": "trazo libre cerrado", "verts": v, "closed": True}
        return {"name": "trazo libre", "verts": rdp(pts, 0.3), "closed": False}

    # ---- Detección de figura ----
    eps = sens_pct / 100.0 * diag
    if not closed:
        v = rdp(pts, eps)
        name = "línea" if len(v) == 2 else f"polilínea ({len(v) - 1} segmentos)"
        return {"name": name, "verts": v, "closed": False}

    v = prune_closed(rdp(pts + [pts[0]], eps)[:-1], eps)

    # ¿Círculo o polígono? Se compara el error de ajuste de ambos modelos.
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    radii = [math.hypot(p[0] - cx, p[1] - cy) for p in pts]
    r = sum(radii) / len(radii)
    circ_err = sum(abs(ri - r) for ri in radii) / len(radii)
    if len(v) >= 3:
        poly_err = sum(min(seg_dist(p, v[i - 1], v[i]) for i in range(len(v)))
                       for p in pts) / len(pts)
    else:
        poly_err = float("inf")

    if circ_err < poly_err and circ_err < 0.12 * r:
        # Círculo ideal, empezando donde empezó el trazo y en el mismo sentido
        a0 = math.atan2(pts[0][1] - cy, pts[0][0] - cx)
        sense = 1 if signed_area(pts) >= 0 else -1
        n = max(24, math.ceil(2 * math.pi * r / max(step, 0.1)))
        v = [(cx + r * math.cos(a0 + sense * 2 * math.pi * i / n),
              cy + r * math.sin(a0 + sense * 2 * math.pi * i / n)) for i in range(n)]
        return {"name": f"círculo (r≈{r:.1f} mm)", "verts": v, "closed": True}

    if len(v) < 3:
        v = rdp(pts + [pts[0]], 0.3)[:-1]
        return {"name": "trazo libre cerrado", "verts": v, "closed": True}
    return {"name": polygon_name(v), "verts": v, "closed": True}


def rnd(x, dec):
    """Redondeo 'mitad hacia arriba' (evita el redondeo bancario de round())."""
    f = 10 ** dec
    val = math.floor(x * f + 0.5) / f
    return int(val) if dec == 0 else val


def interpolate(verts, step=1.0, dec=0):
    """Interpola linealmente entre vértices consecutivos (x,y) o (x,y,z).
    El número de pasos de cada segmento es max(|dx|,|dy|,|dz|)/paso, de modo
    que (0,0,200)->(10,10,200) da [0,0,200],[1,1,200],...,[10,10,200]."""
    out = []

    def add(p):
        q = tuple(rnd(c, dec) for c in p)
        if not out or out[-1] != q:
            out.append(q)

    if not verts:
        return out
    add(verts[0])
    for a, b in zip(verts, verts[1:]):
        d = [cb - ca for ca, cb in zip(a, b)]
        n = max(1, math.ceil(max(abs(c) for c in d) / step - 1e-9))
        for i in range(1, n + 1):
            add(tuple(ca + dc * i / n for ca, dc in zip(a, d)))
    return out


def build_path(figs, W, H, step=1.0, dec=0, z_draw=0.0, lift=1.0):
    """Construye una trayectoria local (u,v,w) sobre el plano YZ.

    u,v son desplazamientos desde Z_min e Y_min; w=0 es lápiz apoyado,
    w=1 es lápiz levantado. Se interpolan los segmentos en u,v.
    """
    polys = []
    for f in figs:
        vs = list(f["verts"])
        if f["closed"]:
            vs.append(vs[0])
        polys.append([(min(max(x, 0.0), W), min(max(y, 0.0), H)) for x, y in vs])
    if not polys:
        return []
    zd = rnd(z_draw, dec)
    zu = rnd(z_draw + lift, dec)

    verts = []
    for k, vs in enumerate(polys):
        vs = [(rnd(x, dec), rnd(y, dec)) for x, y in vs]
        if k > 0:
            px, py, _ = verts[-1]
            verts.append((px, py, zu))              # sube el lápiz
            verts.append((vs[0][0], vs[0][1], zu))  # se desplaza en el aire
        verts.extend((x, y, zd) for x, y in vs)     # baja y dibuja
    return interpolate(verts, step, dec)


def count_lifts(path):
    """Número de transiciones de lápiz apoyado a levantado."""
    if not path:
        return 0
    zd = path[0][2]
    return sum(1 for a, b in zip(path, path[1:]) if abs(a[2] - zd) < 1e-6 and abs(b[2] - zd) >= 1e-6)


def format_path(path, fmt="lineas", dec=0):
    def num(v):
        return str(v) if dec == 0 else f"{v:.{dec}f}"
    if fmt == "csv":
        return "".join(",".join(num(c) for c in p) + "\n" for p in path)
    items = ["[" + ",".join(num(c) for c in p) + "]" for p in path]
    if fmt == "una_linea":
        return ", ".join(items) + "\n"
    return "\n".join(items) + "\n"


def parse_path(text, z_default=200.0):
    """Lee [x,y,z] / [x,y] o 'x,y,z' / 'x,y'. A los puntos 2D se les asigna
    z_default."""
    num = r"(-?\d+(?:\.\d+)?)"
    sep = r"\s*[,;\s]\s*"
    rows = re.findall(r"\[\s*" + num + r"\s*,\s*" + num + r"\s*(?:,\s*" + num + r"\s*)?\]", text)
    if not rows:
        rows = re.findall(r"^\s*" + num + sep + num + r"(?:" + sep + num + r")?\s*$", text, re.M)
    return [(float(x), float(y), float(z) if z else z_default) for x, y, z in rows]


# =============================================================================
#  Coordenadas del robot: X normal al plano, Y vertical, Z horizontal
# =============================================================================

# El lienzo usa coordenadas locales (u,v,w): u crece hacia +Z, v hacia +Y,
# w=0 para dibujar y w=1 para levantar el lapiz. Los TXT usan (X,Y,Z).
LADOS = {"Izquierda": "left", "Derecha": "right"}


def yz_to_xyz(path, z_min, y_min, x_draw, x_lift):
    """Convierte posiciones locales del lienzo a coordenadas XYZ absolutas.

    El lapiz se eleva disminuyendo X (segun convencion del robot).
    """
    return [(float(x_draw) - float(x_lift) * float(w),
             float(y_min) + float(v), float(z_min) + float(u))
            for u, v, w in path]


def format_leg_path(leg, header=""):
    lines = [f"# {h}\n" for h in header.splitlines()]
    lines += [f"{x:.2f}, {y:.2f}, {z:.2f}\n" for x, y, z in leg]
    return "".join(lines)


def load_leg_ik():
    """Importa robot_kinematics.kinem_muneco. Devuelve el módulo, o None."""
    if KINEMATICS_PKG not in sys.path:
        sys.path.insert(0, KINEMATICS_PKG)
    try:
        from robot_kinematics import kinem_muneco
    except Exception:
        return None
    return kinem_muneco


def load_perfiles():
    """Importa robot_kinematics.perfiles_temporales (ley temporal del
    lápiz). Devuelve el módulo, o None."""
    if KINEMATICS_PKG not in sys.path:
        sys.path.insert(0, KINEMATICS_PKG)
    try:
        from robot_kinematics import perfiles_temporales
    except Exception:
        return None
    return perfiles_temporales


LEYES = ("lineal", "cubico", "quintico")


def unreachable_points(leg, ik, side="left"):
    """Índices de los puntos que la pierna no alcanza dentro de los límites
    del control (la misma comprobación que hace la interfaz)."""
    bad = []
    seed = None
    for i, (x, y, z) in enumerate(leg):
        t, ok, _ = ik.ik_geometrico(x, y, z, side=side, semilla_deg=seed)
        if ok:
            seed = t
        else:
            bad.append(i)
    return bad


# =============================================================================
#  Workspace YZ: soluciones analíticas vectorizadas, mismo modelo y límites
#  que kinem_muneco (sin dependencias de ROS 2 ni de actuadores).
# =============================================================================

def workspace_analitico(ik, x_fijo, side, y_axis, z_axis, lift=10.0,
                         margen_deg=5.0, flexion_min_deg=10.0):
    """Devuelve máscara segura, máscara alcanzable y margen para cada Y,Z.

    Se evalúan ambas raíces de la cadera y ambas ramas de rodilla mediante
    expresiones algebraicas; se comprueba además la pose con el lápiz elevado.
    Los márgenes se aplican en las dos alturas, evitando aceptar un punto de
    dibujo que no admita retirar el lápiz verticalmente.
    """
    import numpy as np
    Y, Z = np.meshgrid(np.asarray(y_axis, float), np.asarray(z_axis, float), indexing="ij")
    limits = np.asarray(ik.LIMITES_CONTROL_DEG, dtype=float)

    def level(x):
        pos = ik._equipo_a_modelo(np.stack([np.full_like(Y, x), Y, Z], axis=-1), side)
        xp, yp, zp = ik._a_izquierda((pos[..., 0], pos[..., 1], pos[..., 2]), side)
        reachable = np.zeros(Y.shape, dtype=bool)
        safe = np.zeros(Y.shape, dtype=bool)
        best_margin = np.full(Y.shape, -np.inf)
        # Las cuatro soluciones analíticas algebraicas del modelo del proyecto.
        # Emplear las ecuaciones validadas de kinem_muneco evita discrepancias
        # de signos y de referencias entre las dos piernas.
        with np.errstate(invalid='ignore', divide='ignore'):
            for q1,q2,q3,available in ik._soluciones_geometrico_izq(xp,yp,zp):
                q=ik._q_desde_izquierda((q1,q2,q3),side)
                t=ik.modelo_a_interfaz(q,side)
                margin=np.minimum.reduce([t[j]-limits[j,0] for j in range(3)]+[limits[j,1]-t[j] for j in range(3)])
                valid=available & np.isfinite(margin) & (margin>=-1e-7)
                reachable |= valid
                flexion=np.abs(t[2])
                good=valid & (margin>=margen_deg) & (flexion>=flexion_min_deg)
                safe |= good
                best_margin=np.maximum(best_margin, np.where(valid, margin, -np.inf))
        return reachable, safe, best_margin

    reachable, safe, margin = level(x_fijo)
    if lift > 0:
        _, raised, _ = level(x_fijo-lift)
        safe &= raised
    return reachable, safe, margin


def largest_square(mask, preferred=None):
    """Mayor cuadrado alineado a ejes en una máscara equiespaciada [Y,Z]."""
    import numpy as np
    rows, cols = mask.shape
    dp = np.zeros((rows+1, cols+1), dtype=np.int32)
    best = (0, 0, 0)
    best_dist = float("inf")
    for i in range(1, rows+1):
        for j in range(1, cols+1):
            if mask[i-1,j-1]:
                n=1+min(dp[i-1,j],dp[i,j-1],dp[i-1,j-1])
                dp[i,j]=n
                center = (i-(n+1)/2, j-(n+1)/2)
                d = ((center[0]-preferred[0])**2 + (center[1]-preferred[1])**2) if preferred else 0
                if n > best[0] or (n == best[0] and d < best_dist):
                    best=(int(n), i-n, j-n)
                    best_dist=d
    return best


# =============================================================================
#  Interfaz gráfica
# =============================================================================

FORMATOS = {
    "[X,Y,Z] uno por línea": "lineas",
    "[X,Y,Z], [X,Y,Z], ... en una línea": "una_linea",
    "X,Y,Z (CSV)": "csv",
}


class Trazador(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Trazador – dibujo a trayectoria")
        self.strokes = []      # trazos en mm
        self.current = None
        self.figs = []
        self.path = []
        self.loaded = False    # True si la trayectoria viene de un .txt
        self.sim_job = None
        self._proc_job = None
        self.bad = set()       # índices de puntos fuera de alcance
        self.ik = load_leg_ik()

        # Papel recomendado del muñeco (pierna izquierda) como área inicial.
        papel = (self.ik.PAPEL_RECOMENDADO["left"] if self.ik is not None else
                 {"x_papel": 364.0, "centro": (0.0, -157.0), "lado": 150.0, "levantar": 10.0})
        (y_c, z_c), media = papel["centro"], papel["lado"] / 2
        self.var_ymin = tk.DoubleVar(value=y_c - media)
        self.var_ymax = tk.DoubleVar(value=y_c + media)
        self.var_zmin = tk.DoubleVar(value=z_c - media)
        self.var_zmax = tk.DoubleVar(value=z_c + media)
        self.var_mode = tk.StringVar(value="Detectar figura")
        self.var_sens = tk.DoubleVar(value=6)
        self.var_step = tk.DoubleVar(value=1)
        self.var_dec = tk.IntVar(value=2)
        self.var_close = tk.BooleanVar(value=True)
        self.var_fmt = tk.StringVar(value=list(FORMATOS)[0])
        self.var_show_pts = tk.BooleanVar(value=True)
        self.var_xp = tk.DoubleVar(value=papel["x_papel"])
        self.var_leg_lift = tk.DoubleVar(value=papel["levantar"])
        self.var_side = tk.StringVar(value="Izquierda")
        self.var_check = tk.BooleanVar(value=self.ik is not None)
        self.reach = tk.StringVar(value="")
        self.perf = load_perfiles()
        self.var_ley = tk.StringVar(value=self.perf.NOMBRES["quintico"] if self.perf else "")
        self.var_v_lapiz = tk.DoubleVar(value=20)
        self.var_a_lapiz = tk.DoubleVar(value=100)
        self.var_esquinas = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Dibuja con el botón izquierdo del ratón.")
        self.info = tk.StringVar(value="Sin trayectoria")

        self.ws_x = tk.DoubleVar(value=self.var_xp.get())
        self.ws_side = tk.StringVar(value=self.var_side.get())
        self.ws_resolution = 2.0  # mm / celda; compromiso rapidez vs precisión (muñeco ~430 mm)
        self.ws_summary = tk.StringVar(value="Selecciona X y pulsa Analizar workspace.")
        self.ws_result = None
        self.ws_running = False
        self._build_ui()
        self._set_area()
        for v in (self.var_mode, self.var_sens, self.var_step, self.var_dec,
                  self.var_close, self.var_show_pts, self.var_fmt,
                  self.var_xp, self.var_check, self.var_leg_lift):
            v.trace_add("write", lambda *_: self._schedule_process())

    # ---------------------------------------------------------------- UI ----
    def _build_ui(self):
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True)
        self.tab_dibujo = ttk.Frame(self.tabs)
        self.tab_workspace = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_dibujo, text="Dibujo de trayectoria")
        self.tabs.add(self.tab_workspace, text="Workspace del dibujo")
        main = ttk.Frame(self.tab_dibujo, padding=6)
        main.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(main, bg="#f4f4f4", highlightthickness=0, cursor="pencil")
        self.canvas.grid(row=0, column=0, sticky="nw")
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>", self._on_motion)

        # Panel lateral con barra de desplazamiento: en pantallas bajas no
        # entra completo y "Archivo de salida" y los botones quedaban fuera.
        side_box = ttk.Frame(main)
        side_box.grid(row=0, column=1, sticky="ns")
        main.rowconfigure(0, weight=1)
        side_canvas = tk.Canvas(side_box, highlightthickness=0, width=320)
        side_scroll = ttk.Scrollbar(side_box, orient="vertical", command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_scroll.set)
        side_scroll.pack(side="right", fill="y")
        side_canvas.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(side_canvas, padding=(10, 0))
        side_canvas.create_window((0, 0), window=side, anchor="nw")
        side.bind("<Configure>", lambda e: side_canvas.configure(
            scrollregion=side_canvas.bbox("all"), width=side.winfo_reqwidth()))

        def _rueda(event):
            if event.num == 4 or getattr(event, "delta", 0) > 0:
                side_canvas.yview_scroll(-1, "units")
            else:
                side_canvas.yview_scroll(1, "units")

        def _activar_rueda(_event):
            for sec in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
                side_canvas.bind_all(sec, _rueda)

        def _desactivar_rueda(_event):
            for sec in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
                side_canvas.unbind_all(sec)

        side_box.bind("<Enter>", _activar_rueda)
        side_box.bind("<Leave>", _desactivar_rueda)

        g = ttk.LabelFrame(side, text="Plano YZ — límites absolutos (mm)", padding=6)
        g.pack(fill="x", pady=3)
        for row, (label, variable) in enumerate((
            ("Y mínimo (vertical)", self.var_ymin),
            ("Y máximo (vertical)", self.var_ymax),
            ("Z mínimo (horizontal)", self.var_zmin),
            ("Z máximo (horizontal)", self.var_zmax),
        )):
            ttk.Label(g, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(g, textvariable=variable, width=13).grid(row=row, column=1, sticky="w")
        ttk.Button(g, text="Aplicar rangos", command=self._set_area).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        ttk.Label(g, text="Eje horizontal: Z   |   Eje vertical: Y", foreground="#555").grid(
            row=5, column=0, columnspan=2, sticky="w")

        p = ttk.LabelFrame(side, text="Procesamiento", padding=6)
        p.pack(fill="x", pady=3)
        ttk.Label(p, text="Modo").grid(row=0, column=0, sticky="w")
        ttk.Combobox(p, textvariable=self.var_mode, state="readonly", width=16,
                     values=["Detectar figura", "Trazo libre"]).grid(row=0, column=1, columnspan=2, sticky="w")
        ttk.Label(p, text="Tolerancia esquinas %").grid(row=1, column=0, sticky="w")
        ttk.Scale(p, from_=2, to=15, variable=self.var_sens, length=110).grid(row=1, column=1)
        self.lbl_sens = ttk.Label(p, width=4)
        self.lbl_sens.grid(row=1, column=2)
        self.var_sens.trace_add("write", lambda *_: self.lbl_sens.config(text=f"{self.var_sens.get():.0f}"))
        self.lbl_sens.config(text="6")
        ttk.Label(p, text="Paso interpolación (mm)").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(p, from_=0.1, to=20, increment=0.5, textvariable=self.var_step,
                    width=7).grid(row=2, column=1, sticky="w")
        ttk.Label(p, text="Decimales").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(p, from_=0, to=3, increment=1, textvariable=self.var_dec,
                    width=7).grid(row=3, column=1, sticky="w")
        ttk.Checkbutton(p, text="Cerrar figuras automáticamente",
                        variable=self.var_close).grid(row=4, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(p, text="Mostrar puntos interpolados",
                        variable=self.var_show_pts).grid(row=6, column=0, columnspan=3, sticky="w")

        k = ttk.LabelFrame(side, text="Posición en robot y alcance", padding=6)
        k.pack(fill="x", pady=3)
        ttk.Label(k, text="X constante al dibujar (mm)").grid(row=0, column=0, sticky="w")
        ttk.Entry(k, textvariable=self.var_xp, width=11).grid(row=0, column=1, sticky="w")
        ttk.Label(k, text="Levantamiento del lápiz en X (mm)").grid(row=1, column=0, sticky="w")
        ttk.Entry(k, textvariable=self.var_leg_lift, width=11).grid(row=1, column=1, sticky="w")
        ttk.Label(k, text="Pierna").grid(row=2, column=0, sticky="w")
        ttk.Combobox(k, textvariable=self.var_side, state="readonly", width=11,
                     values=list(LADOS)).grid(row=2, column=1, sticky="w")
        chk = ttk.Checkbutton(k, text="Comprobar alcance de la pierna",
                              variable=self.var_check)
        chk.grid(row=3, column=0, columnspan=2, sticky="w")
        if self.ik is None:
            chk.state(["disabled"])
            self.reach.set("Sin robot_kinematics: comprobación no disponible.")
        self.lbl_reach = ttk.Label(k, textvariable=self.reach, wraplength=280, justify="left")
        self.lbl_reach.grid(row=4, column=0, columnspan=2, sticky="w")

        lt = ttk.LabelFrame(side, text="Ley temporal del lápiz", padding=6)
        lt.pack(fill="x", pady=3)
        if self.perf is None:
            ttk.Label(lt, text="No se encontró robot_kinematics:\nla simulación va a ritmo fijo.").grid(
                row=0, column=0, sticky="w")
        else:
            cb_ley = ttk.Combobox(lt, textvariable=self.var_ley, state="readonly", width=30,
                                  values=[self.perf.NOMBRES[m] for m in LEYES])
            cb_ley.grid(row=0, column=0, columnspan=4, sticky="w")
            cb_ley.bind("<<ComboboxSelected>>", lambda e: self._mostrar_condiciones())
            ttk.Label(lt, text="v máx [mm/s]").grid(row=1, column=0, sticky="w")
            self.sp_v = ttk.Spinbox(lt, from_=1, to=200, increment=5, textvariable=self.var_v_lapiz,
                                    width=6)
            self.sp_v.grid(row=1, column=1, sticky="w")
            ttk.Label(lt, text="a máx [mm/s²]").grid(row=1, column=2, sticky="w")
            self.sp_a = ttk.Spinbox(lt, from_=10, to=2000, increment=10, textvariable=self.var_a_lapiz,
                                    width=6)
            self.sp_a.grid(row=1, column=3, sticky="w")
            ttk.Checkbutton(lt, text="Parar en esquinas y suavizar escalones",
                            variable=self.var_esquinas).grid(row=2, column=0, columnspan=4, sticky="w")
            self.lbl_cond = ttk.Label(lt, wraplength=280, justify="left", foreground="#555")
            self.lbl_cond.grid(row=3, column=0, columnspan=4, sticky="w", pady=(3, 0))
            self._mostrar_condiciones()

        o = ttk.LabelFrame(side, text="Archivo de salida", padding=6)
        o.pack(fill="x", pady=3)
        ttk.Combobox(o, textvariable=self.var_fmt, state="readonly", width=34,
                     values=list(FORMATOS)).pack(fill="x")

        b = ttk.Frame(side)
        b.pack(fill="x", pady=6)
        btns = [("Procesar  (Enter)", self.process), ("Simular trayectoria", self.simulate),
                ("Guardar .txt  (Ctrl+S)", self.save), ("Abrir .txt  (Ctrl+O)", self.open_txt),
                ("Deshacer  (Ctrl+Z)", self.undo), ("Limpiar  (Supr)", self.clear),
                ("Exportar para la pierna  (Ctrl+E)", self.export_leg)]
        for i, (t, c) in enumerate(btns):
            ttk.Button(b, text=t, command=c).grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2,
                                                  columnspan=2 if i == len(btns) - 1 else 1)
        b.columnconfigure((0, 1), weight=1)

        ttk.Label(side, textvariable=self.info, wraplength=280, justify="left",
                  foreground="#1a4f8a").pack(fill="x", pady=3)
        ttk.Label(side, text="Vista previa del archivo:").pack(anchor="w")
        self.preview = tk.Text(side, width=34, height=10, font=("monospace", 9))
        self.preview.pack(fill="both", expand=True)

        ttk.Label(main, textvariable=self.status, relief="sunken", anchor="w").grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        self._build_workspace_ui()
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Control-s>", lambda e: self.save())
        self.bind("<Control-o>", lambda e: self.open_txt())
        self.bind("<Control-e>", lambda e: self.export_leg())
        self.bind("<Return>", lambda e: self.process())
        self.bind("<Delete>", lambda e: self.clear())

    # ---------------------------------------------------- workspace YZ ----
    def _build_workspace_ui(self):
        parent = ttk.Frame(self.tab_workspace, padding=10)
        parent.pack(fill="both", expand=True)
        panel = ttk.Frame(parent)
        panel.pack(side="left", fill="y", padx=(0, 10))
        par = ttk.LabelFrame(panel, text="Parámetros de análisis", padding=8)
        par.pack(fill="x", pady=4)
        ttk.Label(par, text="X fija (mm)").grid(row=0, column=0, sticky="w")
        ttk.Entry(par, textvariable=self.ws_x, width=13).grid(row=0, column=1, sticky="w")
        ttk.Label(par, text="Pierna").grid(row=1, column=0, sticky="w")
        ttk.Combobox(par, textvariable=self.ws_side, state="readonly", width=11,
                     values=list(LADOS)).grid(row=1, column=1, sticky="w")
        ttk.Label(par, text="IK por defecto: algebraico (4 ramas)").grid(row=2, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Label(par, text=f"Resolución: {self.ws_resolution:g} mm").grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Label(par, text="Margen articular ≥ 5°; rodilla ≥ 10°\n"
                              "El lápiz elevado también debe ser alcanzable.",
                  justify="left", foreground="#555").grid(row=4, column=0, columnspan=2, sticky="w")
        self.ws_analyze_btn = ttk.Button(par, text="Analizar workspace", command=self.analyze_workspace)
        self.ws_analyze_btn.grid(row=5, column=0, columnspan=2, sticky="ew", pady=5)
        stats=ttk.LabelFrame(panel, text="Resultados (mm)", padding=8)
        stats.pack(fill="both", expand=True, pady=4)
        ttk.Label(stats, textvariable=self.ws_summary, justify="left", wraplength=300).pack(anchor="w")
        self.ws_apply_btn=ttk.Button(panel, text="Aplicar cuadrado a dibujo", command=self.apply_workspace)
        self.ws_apply_btn.pack(fill="x", pady=10)
        self.ws_apply_btn.state(["disabled"])
        self.ws_canvas=tk.Canvas(parent, width=630, height=615, bg="white", highlightbackground="#bbb", highlightthickness=1)
        self.ws_canvas.pack(side="left", fill="both", expand=True)
        self.ws_canvas.create_text(315, 300, text="Workspace YZ: pulsa «Analizar workspace»", fill="#555")

    def analyze_workspace(self):
        if self.ws_running:
            return
        if self.ik is None:
            messagebox.showerror("Workspace", "No se encontró robot_kinematics.kinem_muneco.\n"
                                 "Abre el trazador desde el proyecto que contiene ros2_ws.")
            return
        try:
            x=float(self.ws_x.get())
            if not math.isfinite(x):
                raise ValueError("X debe ser finita")
            side=LADOS[self.ws_side.get()]
            lift=float(self.var_leg_lift.get())
            if not 0<=lift<=200:
                raise ValueError("Elevación del lápiz fuera de rango [0,200] mm")
        except (ValueError,tk.TclError,KeyError) as err:
            messagebox.showerror("Workspace",str(err))
            return
        self.ws_running=True
        self.ws_analyze_btn.state(["disabled"])
        self.ws_apply_btn.state(["disabled"])
        self.ws_summary.set("Calculando espacio de trabajo y cuadrado seguro...")
        # La IK opera en un hilo; Tk se actualiza solo en hilo principal.
        def worker():
            try:
                import numpy as np
                step=self.ws_resolution
                # Malla a la escala del muñeco (pierna de ~430 mm).
                ya=np.arange(-250., 250.+step*.5,step)
                za=np.arange(-450., 450.+step*.5,step)
                attainable,safe,margin=workspace_analitico(self.ik,x,side,ya,za,lift)
                paper=self.ik.PAPEL_RECOMENDADO[side]['centro']
                prefer=((paper[0]-ya[0])/step,(paper[1]-za[0])/step)
                size,i,j=largest_square(safe, preferred=prefer)
                if size:
                    # Conservar un punto de margen interior por lado: seguro en la
                    # cuadrícula, pero los bordes continuos son una aproximación.
                    margin_cells=1 if size>3 else 0
                    y0,y1=ya[i+margin_cells],ya[i+size-1-margin_cells]
                    z0,z1=za[j+margin_cells],za[j+size-1-margin_cells]
                    square=(float(y0),float(y1),float(z0),float(z1))
                else:
                    square=None
                result=dict(x=x,side=side,step=step,ya=ya,za=za,
                            attainable=attainable,safe=safe,margin=margin,square=square,
                            lift=lift)
                self.after(0,lambda result=result:self._finish_workspace(result,None))
            except Exception as exc:
                self.after(0,lambda msg=str(exc):self._finish_workspace(None,msg))
        threading.Thread(target=worker, daemon=True).start()

    def _finish_workspace(self,result,error):
        self.ws_running=False
        self.ws_analyze_btn.state(["!disabled"])
        if error:
            self.ws_summary.set("No se pudo calcular el workspace: " + error)
            return
        self.ws_result=result
        ya,za=result['ya'],result['za']
        reach=result['attainable']; safe=result['safe']; square=result['square']
        if not reach.any():
            self.ws_summary.set("No hay puntos alcanzables para esta X con los límites actuales.")
        else:
            import numpy as np
            inds=np.where(reach)
            yi=ya[inds[0]]; zi=za[inds[1]]
            txt=(f"X = {result['x']:.2f} mm · {result['side']}\n"
                 f"Alcance total (envolvente):\n"
                 f"Y ∈ [{yi.min():.1f}, {yi.max():.1f}]\n"
                 f"Z ∈ [{zi.min():.1f}, {zi.max():.1f}]\n"
                 f"Ancho Y: {np.ptp(yi):.1f} · Z: {np.ptp(zi):.1f}\n"
                 f"Puntos alcanzables: {reach.sum():,}\n"
                 f"Puntos seguros: {safe.sum():,}\n"
                 f"Malla de {result['step']:g} mm\n")
            if square:
                y0,y1,z0,z1=square
                txt+=(f"\nCUADRADO SEGURO (aprox.):\n"
                      f"Lado: {min(y1-y0,z1-z0):.1f} × {min(y1-y0,z1-z0):.1f} mm\n"
                      f"Centro: Y={(y0+y1)/2:.1f}, Z={(z0+z1)/2:.1f}\n"
                      f"Y ∈ [{y0:.1f}, {y1:.1f}]\n"
                      f"Z ∈ [{z0:.1f}, {z1:.1f}]\n"
                      f"Margen angular ≥5° y rodilla ≥10°\n"
                      f"Incluye posición de lápiz levantado (X−{result['lift']:g}).")
                self.ws_apply_btn.state(["!disabled"])
            else:
                txt+='\nNo se encontró un cuadrado con el margen de seguridad.'
            self.ws_summary.set(txt)
        self._paint_workspace()

    def _paint_workspace(self):
        r=self.ws_result
        if r is None:
            return
        import numpy as np
        from PIL import Image, ImageTk
        c=self.ws_canvas
        c.delete('all')
        ya,za=r['ya'],r['za']; reach=r['attainable']; safe=r['safe']
        # Imagen en píxeles: horizontal Z, vertical Y (ascendente hacia arriba).
        im=np.full((len(ya),len(za),3), (238,238,238), dtype=np.uint8)
        im[reach]=(123,183,220)
        im[safe]=(75,169,112)
        im=np.flipud(im)
        width,height=560,550
        # Image.Resampling existe desde Pillow 9.1; antes la constante está en Image.
        nearest=getattr(Image, 'Resampling', Image).NEAREST
        img=Image.fromarray(im, mode='RGB').resize((width,height),resample=nearest)
        self._ws_image=ImageTk.PhotoImage(img)
        ox,oy=45,35
        c.create_image(ox,oy,image=self._ws_image,anchor='nw')
        c.create_rectangle(ox,oy,ox+width,oy+height,outline='#555')
        def xy(y,z):
            return ox+(z-za[0])/(za[-1]-za[0])*width, oy+(ya[-1]-y)/(ya[-1]-ya[0])*height
        if r['square']:
            y0,y1,z0,z1=r['square']; x0,top=xy(y1,z0); x1,bottom=xy(y0,z1)
            c.create_rectangle(x0,top,x1,bottom,outline='#dc2c23',width=3)
            c.create_text((x0+x1)/2,top-9,text=f"Cuadrado {y1-y0:.0f} mm",fill='#b22222')
        c.create_text(ox+width/2,oy+height+15,text='Z (mm) →',fill='#333')
        c.create_text(12,oy+height/2,text='Y (mm)',angle=90,fill='#333')
        for frac in [0,.25,.5,.75,1]:
            xx=ox+frac*width; zz=za[0]+frac*(za[-1]-za[0])
            c.create_text(xx,oy+height+2,text=f'{zz:.0f}',anchor='n',font=('TkDefaultFont',8))
            yy=ya[0]+frac*(ya[-1]-ya[0]); py=oy+(1-frac)*height
            c.create_text(ox-6,py,text=f'{yy:.0f}',anchor='e',font=('TkDefaultFont',8))
        c.create_text(ox,15,text='Gris: no alcanzable     Azul: alcanzable     Verde: seguro     Rojo: cuadrado',anchor='w')

    def apply_workspace(self):
        r=self.ws_result
        if not r or not r['square']:
            return
        y0,y1,z0,z1=r['square']
        try:
            if (abs(self.ws_x.get()-r['x'])>1e-8 or
                LADOS[self.ws_side.get()] != r['side'] or
                abs(self.var_leg_lift.get()-r['lift'])>1e-8):
                messagebox.showwarning('Workspace', 'Cambiaste X, la pierna o la elevación. Vuelve a analizar antes de aplicar.')
                return
        except (tk.TclError, KeyError):
            messagebox.showerror('Workspace','Parámetros incorrectos.')
            return
        if self.strokes or self.path:
            if not messagebox.askyesno('Aplicar workspace',
                                       'Se limpiará el dibujo o trayectoria actual. ¿Continuar?'):
                return
            self.clear()
        self.var_xp.set(r['x']); self.var_side.set('Izquierda' if r['side']=='left' else 'Derecha')
        self.var_ymin.set(y0); self.var_ymax.set(y1)
        self.var_zmin.set(z0); self.var_zmax.set(z1)
        self._set_area()
        self.tabs.select(self.tab_dibujo)
        self.status.set('Workspace aplicado: cuadrado recomendado en plano YZ.')

    # ------------------------------------------------------- coordenadas ----
    def _set_area(self):
        try:
            ymin, ymax = float(self.var_ymin.get()), float(self.var_ymax.get())
            zmin, zmax = float(self.var_zmin.get()), float(self.var_zmax.get())
            if not all(math.isfinite(t) for t in (ymin, ymax, zmin, zmax)):
                raise ValueError("Los límites deben ser finitos")
            if ymin >= ymax or zmin >= zmax:
                raise ValueError("Cada valor máximo debe superar el mínimo")
        except (tk.TclError, ValueError) as e:
            messagebox.showerror("Plano YZ", f"Rangos no válidos: {e}")
            return
        old = getattr(self, "W", None)
        if old is not None and self.strokes:
            if not messagebox.askyesno("Cambiar rangos", "Los trazos existentes se limpiarán. ¿Continuar?"):
                return
            self.clear()
        self.ymin, self.ymax = ymin, ymax
        self.zmin, self.zmax = zmin, zmax
        self.W, self.H = zmax - zmin, ymax - ymin
        self.scale = CANVAS_MAX_PX / max(self.W, self.H)
        self.canvas.config(width=self.W * self.scale + 2 * PAD,
                           height=self.H * self.scale + 2 * PAD)
        self.process(silent=True)

    def to_px(self, u, v):
        return PAD + u * self.scale, PAD + (self.H - v) * self.scale

    def to_mm(self, px, py):
        u = (px - PAD) / self.scale
        v = self.H - (py - PAD) / self.scale
        return min(max(u, 0.0), self.W), min(max(v, 0.0), self.H)

    def _xyz_path(self):
        return yz_to_xyz(self.path, self.zmin, self.ymin,
                         self.var_xp.get(), self.var_leg_lift.get())

    # ----------------------------------------------------------- dibujo -----
    def _on_press(self, e):
        self._stop_sim()
        if self.loaded:
            self.loaded, self.path, self.figs = False, [], []
            self.redraw()
        self.current = [self.to_mm(e.x, e.y)]

    def _on_drag(self, e):
        if self.current is None:
            return
        p = self.to_mm(e.x, e.y)
        a = self.to_px(*self.current[-1])
        b = self.to_px(*p)
        self.canvas.create_line(*a, *b, width=3, fill="#222", capstyle="round", tags="live")
        self.current.append(p)
        self._on_motion(e)

    def _on_release(self, e):
        if self.current and len(self.current) > 1:
            self.strokes.append(self.current)
        self.current = None
        self.process(silent=True)

    def _on_motion(self, e):
        x, y = self.to_mm(e.x, e.y)
        self.status.set(f"Z = {self.zmin + x:7.2f} mm   Y = {self.ymin + y:7.2f} mm   "
                        f"trazos: {len(self.strokes)}")

    def undo(self):
        if self.strokes:
            self.strokes.pop()
        self.process(silent=True)

    def clear(self):
        self._stop_sim()
        self.strokes, self.figs, self.path, self.loaded = [], [], [], False
        self._check_reach()
        self.redraw()
        self._update_info()

    # ------------------------------------------------------ procesamiento ---
    def _schedule_process(self):
        if self._proc_job:
            self.after_cancel(self._proc_job)
        self._proc_job = self.after(150, lambda: self.process(silent=True))

    def _params(self):
        step = float(self.var_step.get())
        dec = int(self.var_dec.get())
        if not (0.01 <= step <= 1000) or not (0 <= dec <= 6):
            raise ValueError
        return step, dec, 0.0, 1.0

    def process(self, silent=False):
        self._proc_job = None
        if self.loaded:
            self._check_reach()
            self.redraw()
            return
        try:
            step, dec, z, lift = self._params()
        except (tk.TclError, ValueError):
            if not silent:
                messagebox.showerror("Parámetros", "Paso, decimales o Z no válidos")
            return
        mode = "libre" if self.var_mode.get() == "Trazo libre" else "figura"
        self.figs = []
        for s in self.strokes:
            f = analyze_stroke(s, mode, float(self.var_sens.get()),
                               self.var_close.get(), step)
            if f:
                self.figs.append(f)
        self.path = build_path(self.figs, self.W, self.H, step, dec, z, lift)
        self._check_reach()
        self.redraw()
        self._update_info()

    def _side(self):
        return LADOS.get(self.var_side.get(), "left")

    def _check_reach(self):
        self.bad = set()
        if self.ik is None:
            return
        if not self.var_check.get() or not self.path:
            self.reach.set("")
            return
        try:
            leg = self._xyz_path()
        except (tk.TclError, ValueError):
            self.reach.set("Parámetros del marco no válidos")
            self.lbl_reach.config(foreground="#d62728")
            return
        self.bad = set(unreachable_points(leg, self.ik, self._side()))
        if self.bad:
            self.reach.set(f"✗ {len(self.bad)} de {len(leg)} puntos fuera de alcance "
                           "(en rojo). Prueba otra altura, otro centro o una escala menor.")
            self.lbl_reach.config(foreground="#d62728")
        else:
            self.reach.set(f"✓ Los {len(leg)} puntos son alcanzables.")
            self.lbl_reach.config(foreground="#2ca02c")

    def _update_info(self):
        if not self.path:
            self.info.set("Sin trayectoria")
        elif self.loaded:
            self.info.set(f"Archivo XYZ: {len(self.path)} puntos   ·   "
                          f"levantamientos de lápiz: {count_lifts(self.path)}")
        else:
            names = ", ".join(f["name"] for f in self.figs)
            n_v = sum(len(f["verts"]) for f in self.figs if not f["name"].startswith("círculo"))
            self.info.set(f"Figura(s): {names}\nPuntos interpolados: {len(self.path)}"
                          + (f"   ·   vértices: {n_v}" if n_v else "")
                          + f"\nLevantamientos de lápiz: {count_lifts(self.path)}")
        self.preview.delete("1.0", "end")
        if self.path:
            txt = format_path(self._xyz_path()[:400], FORMATOS[self.var_fmt.get()], self._dec())
            if len(self.path) > 400:
                txt += "...\n"
            self.preview.insert("1.0", txt)

    def _dec(self):
        if self.loaded:
            return 0 if all(float(v).is_integer() for p in self.path for v in p) else 3
        try:
            return int(self.var_dec.get())
        except (tk.TclError, ValueError):
            return 0

    # ------------------------------------------------------------ dibujo ----
    def redraw(self):
        c = self.canvas
        c.delete("all")
        x0, y0 = self.to_px(0, self.H)
        x1, y1 = self.to_px(self.W, 0)
        c.create_rectangle(x0, y0, x1, y1, fill="white", outline="#888")
        g = GRID_MM
        while g * self.scale < 25:
            g *= 2
        k = 0
        while k <= self.W + 1e-9:
            px, _ = self.to_px(k, 0)
            c.create_line(px, y0, px, y1, fill="#e6e6e6")
            c.create_text(px, y1 + 12, text=f"{self.zmin + k:g}", fill="#777", font=("sans", 8))
            k += g
        k = 0
        while k <= self.H + 1e-9:
            _, py = self.to_px(0, k)
            c.create_line(x0, py, x1, py, fill="#e6e6e6")
            c.create_text(x0 - 14, py, text=f"{self.ymin + k:g}", fill="#777", font=("sans", 8))
            k += g
        c.create_text(x1, y1 + 26, text="Z (mm)", anchor="e", fill="#555", font=("sans", 8))
        c.create_text(x0 - 4, y0 - 14, text="Y (mm)", anchor="w", fill="#555", font=("sans", 8))
        c.create_rectangle(x0, y0, x1, y1, outline="#888")

        stroke_col = "#c8c8c8" if self.path else "#222"
        for s in self.strokes:
            if len(s) > 1:
                c.create_line(*[v for p in s for v in self.to_px(*p)], width=3,
                              fill=stroke_col, capstyle="round", joinstyle="round")

        if not self.path:
            return
        # Azul: lápiz apoyado; gris discontinuo: desplazamiento con lápiz levantado.
        # Se proyectan los puntos XYZ sobre el plano YZ.
        zd = self.path[0][2]
        pts = [self.to_px(p[0], p[1]) for p in self.path]
        down = [abs(p[2] - zd) < 1e-6 for p in self.path]
        run = [pts[0]]
        for i in range(1, len(pts)):
            seg_down = down[i - 1] and down[i]
            prev_down = down[i - 2] and down[i - 1] if i > 1 else seg_down
            if seg_down != prev_down:
                self._draw_run(run, prev_down)
                run = [pts[i - 1]]
            run.append(pts[i])
        if len(pts) > 1:
            self._draw_run(run, down[-2] and down[-1])
        for i in range(len(pts) - 1):
            if down[i] != down[i + 1]:   # punto donde sube o baja el lápiz
                x, y = pts[i] if down[i] else pts[i + 1]
                c.create_text(x, y - 9, text="▲" if down[i] else "▼",
                              fill="#888", font=("sans", 8))
        if self.var_show_pts.get() and len(pts) < 5000:
            for (x, y), d in zip(pts, down):
                if d:
                    c.create_oval(x - 1.5, y - 1.5, x + 1.5, y + 1.5, fill="#1f6fd1", outline="")
        if not self.loaded:
            for f in self.figs:
                if not f["name"].startswith("círculo"):
                    for v in f["verts"]:
                        x, y = self.to_px(*v)
                        c.create_oval(x - 4, y - 4, x + 4, y + 4, outline="#d62728", width=2)
                ys = [v[1] for v in f["verts"]]
                xs = [v[0] for v in f["verts"]]
                tx, ty = self.to_px(sum(xs) / len(xs), max(ys))
                c.create_text(tx, ty - 12, text=f["name"], fill="#d62728", font=("sans", 9, "bold"))
        for i in self.bad:
            x, y = pts[i]
            c.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#d62728", outline="")
        sx, sy = pts[0]
        c.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, fill="#2ca02c", outline="")
        c.create_text(sx + 8, sy + 8, text="inicio", anchor="nw", fill="#2ca02c", font=("sans", 8))

    def _draw_run(self, run, pen_down):
        if len(run) < 2:
            return
        flat = [v for p in run for v in p]
        if pen_down:
            self.canvas.create_line(*flat, fill="#1f6fd1", width=2)
        else:
            self.canvas.create_line(run[0][0], run[0][1], run[-1][0], run[-1][1],
                                    fill="#999", width=1, dash=(5, 4))

    # -------------------------------------------------------- simulación ----
    def _stop_sim(self):
        if self.sim_job:
            self.after_cancel(self.sim_job)
            self.sim_job = None
        self.canvas.delete("sim")

    def _ley(self):
        inv = {v: k for k, v in self.perf.NOMBRES.items()}
        return inv.get(self.var_ley.get(), "quintico")

    def _mostrar_condiciones(self):
        """Qué fija la ley elegida y qué parámetros usa (los demás se
        desactivan)."""
        texto, usa = self.perf.CONDICIONES_CAMINO[self._ley()]
        self.lbl_cond.config(text=texto)
        self.sp_v.state(["!disabled"] if "v" in usa else ["disabled"])
        self.sp_a.state(["!disabled"] if "a" in usa else ["disabled"])

    def _timeline(self):
        """Muestras (cada 20 ms) del recorrido del lápiz con la ley temporal
        elegida: (posiciones Nx3 en el papel, lápiz abajo N)."""
        P = [tuple(p) for p in self.path]
        zd = P[0][2]
        metodo, v, a = self._ley(), float(self.var_v_lapiz.get()), float(self.var_a_lapiz.get())
        esquinas = self.var_esquinas.get()
        pos, abajo = [], []
        i = 0
        while i < len(P):
            j = i
            down = P[i][2] == zd
            while j + 1 < len(P) and (P[j + 1][2] == zd) == down:
                j += 1
            # el tramo incluye el punto de unión con el siguiente
            tramo = P[i:j + 2] if j + 1 < len(P) else P[i:j + 1]
            if len(tramo) >= 2:
                lados = [tramo]
                if down and esquinas:
                    c = self.perf.dividir_en_esquinas(tramo)
                    lados = [self.perf.suavizar_lado(tramo[x:y + 1]) for x, y in zip(c, c[1:])]
                for lado in lados:
                    _, muestras = self.perf.ley_temporal_camino(lado, metodo, v, a, 0.02)
                    pos.extend(muestras)
                    abajo.extend([down] * len(muestras))
            i = j + 1
        return pos, abajo

    def simulate(self):
        self._stop_sim()
        if not self.path:
            return
        if self.perf is None:
            return self._simulate_fijo()
        try:
            pos, abajo = self._timeline()
        except (self.perf.ErrorPerfil, ValueError, tk.TclError) as error:
            messagebox.showerror("Simular", f"Ley temporal no válida: {error}")
            return
        pts = [self.to_px(p[0], p[1]) for p in pos]
        T = 0.02 * (len(pts) - 1)
        t_abajo = 0.02 * sum(abajo)
        t0 = time.monotonic()
        estado = {"i": 0}

        def step():
            i_obj = min(len(pts) - 1, int((time.monotonic() - t0) / 0.02))
            i = estado["i"]
            while i < i_obj:
                if abajo[i] and abajo[i + 1]:
                    self.canvas.create_line(*pts[i], *pts[i + 1], fill="#ff7f0e", width=4,
                                            capstyle="round", tags="sim")
                i += 1
            estado["i"] = i
            self.canvas.delete("tip")
            x, y = pts[i]
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6,
                                    fill="#ff7f0e" if abajo[i] else "",
                                    outline="black", width=1 if abajo[i] else 2, tags=("sim", "tip"))
            self.status.set(f"Simulando ({self.var_ley.get()}): {0.02 * i:5.1f} / {T:.1f} s"
                            f"   ·   lápiz abajo {t_abajo:.1f} s"
                            + ("" if abajo[i] else "   (lápiz arriba)"))
            if i >= len(pts) - 1:
                self.sim_job = None
                return
            self.sim_job = self.after(20, step)
        step()

    def _simulate_fijo(self):
        path = self.path
        zd = path[0][2]
        pts = [self.to_px(p[0], p[1]) for p in path]
        delay = max(1, min(20, 4000 // len(pts)))

        def step(i):
            if i >= len(pts):
                self.sim_job = None
                return
            down = path[i][2] == zd
            if i > 0 and down and path[i - 1][2] == zd:
                self.canvas.create_line(*pts[i - 1], *pts[i], fill="#ff7f0e", width=4,
                                        capstyle="round", tags="sim")
            self.canvas.delete("tip")
            x, y = pts[i]
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6,
                                    fill="#ff7f0e" if down else "",
                                    outline="black", width=1 if down else 2, tags=("sim", "tip"))
            self.status.set(f"Simulando punto {i + 1}/{len(path)}:  "
                            + ", ".join(f"{c:g}" for c in path[i])
                            + ("" if down else "   (lápiz arriba)"))
            self.sim_job = self.after(delay, step, i + 1)
        step(0)

    # --------------------------------------------------------- archivos -----
    def save(self):
        if not self.loaded:
            self.process(silent=True)
        if not self.path:
            messagebox.showinfo("Guardar", "No hay ninguna trayectoria que guardar.")
            return
        fn = filedialog.asksaveasfilename(defaultextension=".txt",
                                          filetypes=[("Texto", "*.txt"), ("Todos", "*")],
                                          initialfile="trayectoria.txt")
        if not fn:
            return
        with open(fn, "w", encoding="utf-8") as fh:
            fh.write(format_path(self._xyz_path(), FORMATOS[self.var_fmt.get()], self._dec()))
        self.status.set(f"Guardado {fn}  ({len(self.path)} puntos)")

    def open_txt(self):
        fn = filedialog.askopenfilename(filetypes=[("Texto", "*.txt"), ("Todos", "*")])
        if not fn:
            return
        try:
            with open(fn, encoding="utf-8") as fh:
                xyz = parse_path(fh.read(), 0.0)
            if not xyz:
                raise ValueError("No se encontraron puntos XYZ válidos.")
            if not all(all(math.isfinite(a) for a in p) for p in xyz):
                raise ValueError("Archivo con números no finitos")
            x0 = xyz[0][0]
            lift = max(0, max(x0 - p[0] for p in xyz))
            if lift > 0:
                self.var_leg_lift.set(lift)
            self.var_xp.set(x0)
            yvals = [p[1] for p in xyz]
            zvals = [p[2] for p in xyz]
            if (min(yvals) < self.ymin or max(yvals) > self.ymax or
                    min(zvals) < self.zmin or max(zvals) > self.zmax):
                self.var_ymin.set(min(self.ymin, math.floor(min(yvals))))
                self.var_ymax.set(max(self.ymax, math.ceil(max(yvals))))
                self.var_zmin.set(min(self.zmin, math.floor(min(zvals))))
                self.var_zmax.set(max(self.zmax, math.ceil(max(zvals))))
                self.strokes.clear()
                self._set_area()
            self._stop_sim()
            lift_use = self.var_leg_lift.get()
            path = [(z - self.zmin, y - self.ymin,
                     (x0 - x) / lift_use if lift_use > 0 else 0.0)
                    for x, y, z in xyz]
            self.strokes, self.figs, self.path, self.loaded = [], [], path, True
            self._check_reach()
            self.redraw()
            self._update_info()
            self.status.set(f"Cargado XYZ: {fn}")
        except (OSError, ValueError, tk.TclError) as err:
            messagebox.showerror("Abrir TXT", str(err))

    def export_leg(self):
        if not self.loaded:
            self.process(silent=True)
        if not self.path:
            messagebox.showinfo("Exportar", "No hay trayectoria que exportar.")
            return
        if self.bad:
            messagebox.showwarning("Exportar", "Hay puntos fuera del espacio de trabajo. "
                                   "Revisa los límites antes de utilizar la trayectoria.")
            return
        fn = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Texto", "*.txt"), ("Todos", "*")],
            initialdir=LEG_TRAJ_DIR if os.path.isdir(LEG_TRAJ_DIR) else None,
            initialfile="trayectoria_pierna.txt")
        if not fn:
            return
        try:
            xyz = self._xyz_path()
            ley = ""
            if self.perf is not None:
                ley = (f"\nley_temporal metodo={self._ley()} v={self.var_v_lapiz.get():g} "
                       f"a={self.var_a_lapiz.get():g} "
                       f"esquinas={'si' if self.var_esquinas.get() else 'no'}")
            header = ("Trayectoria cartesiana XYZ (mm) - lienzo YZ, X fijo al dibujar\n"
                      f"rangos Y=[{self.ymin:g},{self.ymax:g}], Z=[{self.zmin:g},{self.zmax:g}] "
                      f"X_dibujo={self.var_xp.get():g}, elevacion_X={self.var_leg_lift.get():g}"
                      + ley)
            with open(fn, "w", encoding="utf-8") as fh:
                fh.write(format_leg_path(xyz, header))
            self.status.set(f"Exportado XYZ {fn} ({len(xyz)} puntos)")
        except (OSError, ValueError, tk.TclError) as err:
            messagebox.showerror("Exportar", str(err))


if __name__ == "__main__":
    Trazador().mainloop()

