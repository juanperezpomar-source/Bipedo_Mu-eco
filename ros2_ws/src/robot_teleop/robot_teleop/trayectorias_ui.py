# ============================================================
# PESTAÑA "GENERACIÓN DE TRAYECTORIAS" DE LA INTERFAZ
# ============================================================
#
# Ejecuta las trayectorias dibujadas en el trazador
# (scripts/trazador/trazador.py): calcula cómo se mueve cada
# articulación en el tiempo, muestra posición, velocidad y
# aceleración de cada articulación, y ejecuta el movimiento
# enviando la consigna a 50 Hz (RViz y, opcionalmente, motores).
#
#   - Pose de la pierna: muestra la posición y los ángulos actuales,
#     y lleva la pierna a Home o a la pose de preparación con un
#     movimiento articular suave.
#   - Dibujo: ley temporal cartesiana en los trazos (el lápiz
#     recorre la letra con el perfil elegido) y movimientos
#     articulares con el lápiz levantado; o bien interpolación
#     articular por puntos intermedios.
#
# Los movimientos punto a punto a una coordenada o a unos ángulos
# cualesquiera se hacen desde las pestañas de cinemática.
#
# Las posiciones están en el sistema de las pestañas de cinemática
# inversa (X a lo largo de la pierna hacia abajo, Y adelante, Z
# lateral). Los cálculos están en
# robot_kinematics/perfiles_temporales.py y kinem_muneco.py (el
# equivalente para el muñeco del kinem_v6.py de la pata del parcial).
# ============================================================

import os
import time

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog

from geometry_msgs.msg import Point

from robot_kinematics import perfiles_temporales as pt
from robot_kinematics.kinem_muneco import (
    ik_geometrico,
    posicion_pie,
    pose_preparacion,
    LIMITES_CONTROL_DEG,
    PAPEL_RECOMENDADO,
)
# El rastro de RViz usa la cinemática directa del equipo (marco DH,
# publicado sobre Base_link), igual que las pestañas de IK.
from robot_kinematics.cinematica_directa_der_MUNECO_izq import (
    forward_kinematics_right,
    forward_kinematics_left,
    get_position,
)

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAY_MATPLOTLIB = True
except Exception:      # la pestaña funciona igual, sin gráficas
    HAY_MATPLOTLIB = False


DT = 0.02                       # 50 Hz
NOMBRES_ART = ('roll', 'pitch', 'rodilla')
COLORES = ('#1f77b4', '#d62728', '#2ca02c')

# Techo real del firmware de la ESP32 del muñeco (rampa VEL_DEG_S en
# firmware/muneco_motores/src/main.cpp): 30 °/s. Se usa solo para el
# aviso de _mostrar(); el valor por defecto del campo "v art." queda en
# 15°/s, bastante por debajo, para no recortar el pico de la curva.
VEL_FIRMWARE_DEG_S = 30.0

METODOS_LEY = ('lineal', 'cubico', 'quintico')


def _float(entry, defecto=None):
    try:
        return float(entry.get().replace(',', '.'))
    except (ValueError, tk.TclError):
        if defecto is None:
            raise ValueError
        return defecto


class PestanaTrayectorias:

    def __init__(self, parent, nodo, archivo_por_defecto):

        self.nodo = nodo
        self.side = nodo.leg_side
        self.archivo = archivo_por_defecto
        self.plan = None
        self.after_id = None
        self.t_inicio = None
        self.tick_n = 0
        self.lapiz_abajo_antes = False

        cont = ttk.Frame(parent)
        cont.pack(fill='both', expand=True)

        izq = ttk.Frame(cont)
        izq.pack(side='left', fill='y', padx=(0, 6))
        der = ttk.Frame(cont)
        der.pack(side='left', fill='both', expand=True)

        self._construir_controles(izq)
        self._construir_grafica(der)
        self._actualizar_campos()
        self.estado("Elige el archivo del trazador y pulsa Calcular y graficar.")
        self._refrescar_actual()

    # ========================================================
    # CONTROLES
    # ========================================================

    def _construir_controles(self, f):

        # ---------------- 1. Pose de la pierna ----------------
        q = ttk.LabelFrame(f, text="1. Pose de la pierna", padding=6, style='Section.TLabelframe')
        q.pack(fill='x', pady=(0, 6))

        # Posición y ángulos de la pose actual y del destino (solo lectura:
        # para mover la pierna a otro punto están las pestañas de cinemática).
        mono = ('monospace', 8)
        ttk.Label(q, text="            X       Y       Z [mm]    roll  pitch rodilla [°]",
                  font=mono).grid(row=0, column=0, columnspan=4, sticky='w')
        self.lbl_actual = ttk.Label(q, font=mono)
        self.lbl_actual.grid(row=1, column=0, columnspan=4, sticky='w')
        self.lbl_destino = ttk.Label(q, font=mono, text=f"{'Destino':<8s}   —")
        self.lbl_destino.grid(row=2, column=0, columnspan=4, sticky='w')

        bq = ttk.Frame(q)
        bq.grid(row=3, column=0, columnspan=4, sticky='w', pady=(3, 0))
        ttk.Button(bq, text="Pose de preparación", command=self._ir_a_preparacion).pack(side='left')
        ttk.Button(bq, text="Home", command=self._ir_a_home).pack(side='left', padx=3)

        # ---------------- 2. Dibujo ----------------
        d = ttk.LabelFrame(f, text="2. Dibujo (archivo del trazador)", padding=6,
                           style='Section.TLabelframe')
        d.pack(fill='x', pady=(0, 6))
        fa = ttk.Frame(d)
        fa.grid(row=0, column=0, columnspan=4, sticky='w')
        ttk.Button(fa, text="Archivo...", command=self._elegir_archivo).pack(side='left')
        self.lbl_archivo = ttk.Label(fa, text=os.path.basename(self.archivo), width=26)
        self.lbl_archivo.pack(side='left', padx=4)
        self.var_modo_dibujo = tk.StringVar(value='Ley temporal cartesiana + articular')
        # Modo fijo: ley temporal cartesiana + articular.

        # ---------------- 3. Movimientos articulares ----------------
        # Aproximación al dibujo, desplazamientos con el lápiz levantado y
        # los botones Home / Pose de preparación. Su duración sale de la
        # velocidad articular.
        a = ttk.LabelFrame(f, text="3. Movimientos articulares (lápiz arriba, Home, preparación)",
                           padding=6, style='Section.TLabelframe')
        a.pack(fill='x', pady=(0, 6))

        self.var_metodo = tk.StringVar(value=pt.NOMBRES['quintico'])
        cb = ttk.Combobox(a, textvariable=self.var_metodo, state='readonly', width=34,
                          values=[pt.NOMBRES[m] for m in METODOS_LEY])
        cb.grid(row=0, column=0, columnspan=4, sticky='w')
        cb.bind('<<ComboboxSelected>>', lambda e: self._actualizar_campos())

        self.campos = {}

        def campo(clave, texto, fila, col, valor):
            ttk.Label(a, text=texto).grid(row=fila, column=col, sticky='w', pady=1)
            e = ttk.Entry(a, width=7, justify='center')
            e.insert(0, valor)
            e.grid(row=fila, column=col + 1, sticky='w', padx=(2, 8))
            self.campos[clave] = e

        campo('v_art', "v art. [°/s]", 1, 0, "15")
        self.var_coord = tk.BooleanVar(value=True)
        # Coordinacion isocrona activa por defecto, sin opcion de desactivarla.

        # ---------------- 3. Ley temporal del lápiz ----------------
        l = ttk.LabelFrame(f, text="4. Ley temporal del lápiz (trazos)", padding=6,
                           style='Section.TLabelframe')
        l.pack(fill='x', pady=(0, 6))
        self.l_frame = l
        self.var_ley = tk.StringVar(value=pt.NOMBRES['quintico'])
        cb_ley = ttk.Combobox(l, textvariable=self.var_ley, state='readonly', width=34,
                              values=[pt.NOMBRES[m] for m in METODOS_LEY])
        cb_ley.grid(row=0, column=0, columnspan=4, sticky='w')
        cb_ley.bind('<<ComboboxSelected>>', lambda e: self._actualizar_campos())
        ttk.Label(l, text="v máx [mm/s]").grid(row=1, column=0, sticky='w')
        self.e_v_lapiz = ttk.Entry(l, width=7, justify='center')
        self.e_v_lapiz.insert(0, "20")
        self.e_v_lapiz.grid(row=1, column=1, sticky='w', padx=(2, 8))
        ttk.Label(l, text="a máx [mm/s²]").grid(row=1, column=2, sticky='w')
        self.e_a_lapiz = ttk.Entry(l, width=7, justify='center')
        self.e_a_lapiz.insert(0, "100")
        self.e_a_lapiz.grid(row=1, column=3, sticky='w')
        self.var_esquinas = tk.BooleanVar(value=True)
        ttk.Checkbutton(l, text="Parar en esquinas de más de", variable=self.var_esquinas
                        ).grid(row=2, column=0, columnspan=3, sticky='w')
        self.e_esquina = ttk.Entry(l, width=5, justify='center')
        self.e_esquina.insert(0, "30")
        self.e_esquina.grid(row=2, column=3, sticky='w')
        self.var_suavizar = tk.BooleanVar(value=True)
        ttk.Checkbutton(l, text="Suavizar escalones del redondeo (≤ 0.5 mm)",
                        variable=self.var_suavizar).grid(row=3, column=0, columnspan=4, sticky='w')
        self.lbl_cond = ttk.Label(l, wraplength=330, justify='left', foreground='#555')
        self.lbl_cond.grid(row=4, column=0, columnspan=4, sticky='w', pady=(3, 0))

        # ---------------- Botones ----------------
        b = ttk.Frame(f)
        b.pack(fill='x', pady=(0, 2))
        ttk.Button(b, text="Calcular y graficar", command=self.calcular).pack(side='left')
        ttk.Button(b, text="Ejecutar dibujo", command=self.ejecutar).pack(side='left', padx=3)
        tk.Button(b, text="Parar", bg='#ffdddd', command=self.parar).pack(side='left')
        tk.Button(b, text="Limpiar rastro (RViz)", bg='#ddeeff',
                  command=self.nodo.clear_trail).pack(side='left', padx=(3, 0))
        self.var_hw = tk.BooleanVar(value=False)
        tk.Checkbutton(b, text="+ motores", variable=self.var_hw).pack(side='left', padx=(6, 0))

    def _construir_grafica(self, f):
        self.lbl_estado = tk.Label(f, text="",
                                   anchor='w', justify='left', wraplength=740,
                                   font=('Arial', 9, 'bold'))
        self.lbl_estado.pack(fill='x')
        self.txt = tk.Text(f, height=7, width=60, font=('monospace', 8), wrap='none')
        self.txt.pack(fill='x')
        if not HAY_MATPLOTLIB:
            ttk.Label(f, text="(matplotlib no disponible: sin gráficas)").pack()
            self.fig = None
            return
        self.fig = Figure(figsize=(5.6, 5.2), dpi=90)
        self.ax = [self.fig.add_subplot(3, 1, k + 1) for k in range(3)]
        self.fig.subplots_adjust(left=0.13, right=0.98, top=0.95, bottom=0.08, hspace=0.35)
        self.canvas = FigureCanvasTkAgg(self.fig, master=f)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.cursor = []
        self._dibujar_vacio()

    def _dibujar_vacio(self):
        for a, tit in zip(self.ax, ('Posición q [°]', 'Velocidad q̇ [°/s]', 'Aceleración q̈ [°/s²]')):
            a.clear()
            a.set_ylabel(tit, fontsize=8)
            a.tick_params(labelsize=7)
            a.grid(alpha=0.3)
        self.ax[-1].set_xlabel('t [s]', fontsize=8)
        self.canvas.draw_idle()

    # ========================================================
    # HABILITAR CAMPOS SEGÚN MÉTODO Y MODO
    # ========================================================

    def _metodo(self):
        inv = {v: k for k, v in pt.NOMBRES.items()}
        return inv[self.var_metodo.get()]

    def _ley(self):
        inv = {v: k for k, v in pt.NOMBRES.items()}
        return inv[self.var_ley.get()]

    def _actualizar_campos(self):
        ley_cartesiana = self.var_modo_dibujo.get().startswith('Ley')

        # Ley temporal del lápiz: solo con ley cartesiana; en los modos de
        # puntos intermedios solo cuenta la velocidad del lápiz. v y a se
        # activan si la ley elegida las usa.
        texto, usa_ley = pt.CONDICIONES_CAMINO[self._ley()]
        self.lbl_cond.config(text=texto if ley_cartesiana else '')
        for w in self.l_frame.winfo_children():
            if isinstance(w, ttk.Label):
                continue
            if w is self.e_v_lapiz:
                activo = 'v' in usa_ley if ley_cartesiana else True
            elif w is self.e_a_lapiz:
                activo = ley_cartesiana and 'a' in usa_ley
            else:
                activo = ley_cartesiana
            estado = ('readonly' if isinstance(w, ttk.Combobox) else 'normal') if activo else 'disabled'
            try:
                w.config(state=estado)
            except tk.TclError:
                pass

    # ========================================================
    # POSE DE LA PIERNA: ACTUAL, HOME Y PREPARACIÓN
    # ========================================================

    @staticmethod
    def _fila(nombre, t_deg, side):
        x, y, z = posicion_pie(t_deg, side)
        r, p, k = t_deg
        return f"{nombre:<8s}{x:7.1f} {y:7.1f} {z:7.1f}   {r:6.1f} {p:6.1f} {k:6.1f}"

    def _refrescar_actual(self):
        """La pose actual cambia también desde las otras pestañas: se
        refresca cada 200 ms."""
        try:
            self.lbl_actual.config(text=self._fila("Actual", self.nodo.target_deg, self.side))
            self.nodo.root.after(200, self._refrescar_actual)
        except tk.TclError:          # ventana cerrada
            pass

    def _ir_a_preparacion(self):
        t, ok, motivo = ik_geometrico(*pose_preparacion(self.side), side=self.side,
                                      semilla_deg=self.nodo.target_deg)
        if not ok:
            self.estado(f"✗ Pose de preparación {motivo}", 'red')
            return
        levantar = PAPEL_RECOMENDADO[self.side]['levantar']
        self._ir_a(t, f"la pose de preparación ({levantar:g} mm sobre el centro del papel)")

    def _ir_a_home(self):
        self._ir_a((0.0, 0.0, 0.0), "Home (pierna colgando)")

    def _ir_a(self, t_deg, nombre):
        """Lleva la pierna a t_deg con un movimiento articular suave (el
        perfil de la sección 3), lo grafica y lo ejecuta."""
        self.parar(silencioso=True)
        destino = [float(v) for v in t_deg]
        self.lbl_destino.config(text=self._fila("Destino", destino, self.side))
        q0 = np.asarray(self.nodo.target_deg, float)
        if np.allclose(q0, destino, atol=0.05):
            self.estado(f"La pierna ya está en {nombre}.", 'green')
            return
        try:
            m, p = self._params_articulares()
            t, Q = pt.ptp_muestreado(q0, destino, m, _float(self.campos['v_art']), p,
                                     True, DT)
        except (pt.ErrorPerfil, ValueError, KeyError) as error:
            self.estado(f"✗ {error or 'revisa los parámetros (números)'}", 'red')
            return
        P = np.array([posicion_pie(q, self.side) for q in Q])
        plan = pt.Plan(t, Q, P, np.zeros(len(t), bool),
                       [(float(t[0]), float(t[-1]), f"{pt.NOMBRES[m]} hasta {nombre}")], DT)
        V, A = plan.derivadas()
        self.plan = plan
        self._mostrar(plan, V, A, f"{t[0]:7.2f}-{t[-1]:7.2f} s  {pt.NOMBRES[m]} hasta {nombre}")
        self._iniciar(f"Yendo a {nombre}...")

    def _elegir_archivo(self):
        carpeta = os.path.dirname(self.archivo) if self.archivo else None
        path = filedialog.askopenfilename(parent=self.nodo.root, title="Elegir trayectoria",
                                          initialdir=carpeta,
                                          filetypes=[("Texto", "*.txt"), ("Todos", "*")])
        if path:
            self.archivo = path
            self.lbl_archivo.config(text=os.path.basename(path))
            self._leer_ley_del_archivo(path)

    def _leer_ley_del_archivo(self, path):
        """Si el archivo lo exportó el trazador con una ley temporal,
        se proponen sus parámetros."""
        try:
            with open(path, encoding='utf-8') as fh:
                for linea in fh:
                    if not linea.startswith('#'):
                        break
                    if 'ley_temporal' in linea:
                        datos = dict(par.split('=') for par in linea.split()[2:] if '=' in par)
                        if datos.get('metodo') in METODOS_LEY:
                            self.var_ley.set(pt.NOMBRES[datos['metodo']])
                        for clave, e in (('v', self.e_v_lapiz), ('a', self.e_a_lapiz)):
                            if clave in datos:
                                e.delete(0, tk.END)
                                e.insert(0, datos[clave])
                        if 'esquinas' in datos:
                            activo = datos['esquinas'] == 'si'
                            self.var_esquinas.set(activo)
                            self.var_suavizar.set(activo)
                        self.estado(f"Ley temporal del trazador cargada: {linea[1:].strip()}", 'blue')
        except OSError:
            pass

    def estado(self, texto, color='black'):
        self.lbl_estado.config(text=texto, foreground=color)

    # ========================================================
    # CÁLCULO
    # ========================================================

    def _params_articulares(self):
        """Método de los movimientos articulares. La duración sale de la
        velocidad articular (v_art)."""
        return self._metodo(), {}

    def _plan_dibujo(self):
        from robot_teleop.teleop_node import _load_trajectory_file
        puntos = np.array([(x, y, z) for _, x, y, z in _load_trajectory_file(self.archivo)])
        if len(puntos) < 2:
            raise pt.ErrorPerfil("el archivo no tiene suficientes puntos")
        q0 = list(self.nodo.target_deg)
        m, p = self._params_articulares()
        v_art = _float(self.campos['v_art'])
        modo = self.var_modo_dibujo.get()
        if modo.startswith('Ley'):
            plan = pt.planificar_dibujo(
                puntos, self.side, q0, ik_geometrico, posicion_pie,
                metodo_camino=self._ley(), v_lapiz=_float(self.e_v_lapiz),
                a_lapiz=_float(self.e_a_lapiz), metodo_articular=m, v_articular=v_art,
                params_articular=p, coordinado=True, dt=DT,
                parar_en_esquinas=self.var_esquinas.get(),
                umbral_esquina_deg=_float(self.e_esquina, 30.0),
                suavizar=self.var_suavizar.get())
        else:
            plan = pt.planificar_por_puntos(
                puntos, self.side, q0, ik_geometrico, posicion_pie, 'puntos_cubico',
                v_lapiz=_float(self.e_v_lapiz),
                params={'v_articular': v_art}, dt=DT)
        V, A = plan.derivadas()
        tramos = "\n".join(f"{a:7.2f}-{b:7.2f} s  {d}" for a, b, d in plan.tramos[:40])
        if len(plan.tramos) > 40:
            tramos += f"\n... ({len(plan.tramos)} tramos en total)"
        return plan, (V, A), tramos

    def calcular(self):
        self.parar(silencioso=True)
        try:
            plan, (V, A), texto = self._plan_dibujo()
        except (pt.ErrorPerfil, ValueError, KeyError, OSError) as error:
            self.plan = None
            self.estado(f"✗ {error or 'revisa los parámetros (números)'}", 'red')
            return None
        self.plan = plan
        self._mostrar(plan, V, A, texto)
        return plan

    def _mostrar(self, plan, V, A, texto):
        lim = np.array(LIMITES_CONTROL_DEG)
        vmax = np.abs(V).max(axis=0)
        amax = np.abs(A).max(axis=0)
        avisos = []
        fuera = (plan.angulos < lim[:, 0] - 1e-6) | (plan.angulos > lim[:, 1] + 1e-6)
        if fuera.any():
            avisos.append("✗ hay ángulos fuera de los límites del control")
        if vmax.max() > pt.VEL_MAX_SERVO_DEG_S:
            avisos.append(f"✗ supera la velocidad del servo ({pt.VEL_MAX_SERVO_DEG_S:.0f} °/s)")
        if vmax.max() > VEL_FIRMWARE_DEG_S:
            avisos.append(f"⚠ con motores, el firmware limita a ~{VEL_FIRMWARE_DEG_S:g} °/s: irán más lentos")
        if plan.lapiz_abajo.any():
            altura = plan.posiciones @ np.asarray(pt.ARRIBA)
            if altura[0] < altura[plan.lapiz_abajo].min() - 1.0:
                avisos.append("⚠ la pierna empieza POR DEBAJO del papel: llévala antes a la pose "
                              "de preparación (sin la mesa)")
        resumen = (f"Duración {plan.duracion:.2f} s · {len(plan.t)} muestras a {1 / DT:.0f} Hz\n"
                   + "Vel. máx [°/s]:  " + "  ".join(f"{n} {v:6.1f}" for n, v in zip(NOMBRES_ART, vmax)) + "\n"
                   + "Acel. máx [°/s²]: " + "  ".join(f"{n} {a:6.0f}" for n, a in zip(NOMBRES_ART, amax)))
        self.txt.delete('1.0', tk.END)
        self.txt.insert('1.0', resumen + "\n" + texto)
        if avisos:
            self.estado("\n".join(avisos), 'red' if any(a.startswith('✗') for a in avisos) else 'dark orange')
        else:
            self.estado("✓ Listo para ejecutar.", 'green')
        if self.fig is None:
            return
        self._dibujar_vacio()
        for k, serie in enumerate((plan.angulos, V, A)):
            for j in range(3):
                self.ax[k].plot(plan.t, serie[:, j], color=COLORES[j], lw=1.2, label=NOMBRES_ART[j])
            if plan.lapiz_abajo.any():
                y0, y1 = self.ax[k].get_ylim()
                self.ax[k].fill_between(plan.t, y0, y1, where=plan.lapiz_abajo, color='#999', alpha=0.15,
                                        lw=0, label='lápiz abajo' if k == 0 else None)
                self.ax[k].set_ylim(y0, y1)
        self.ax[0].legend(fontsize=7, loc='best', ncol=4)
        self.cursor = [a.axvline(plan.t[0], color='k', lw=0.8) for a in self.ax]
        self.canvas.draw_idle()

    # ========================================================
    # EJECUCIÓN (50 Hz)
    # ========================================================

    def ejecutar(self):
        plan = self.calcular()
        if plan is None:
            return
        if np.any(np.abs(plan.angulos[0] - np.array(self.nodo.target_deg)) > 0.5):
            self.estado("✗ El plan no empieza en la pose actual; vuelve a calcular.", 'red')
            return
        self.nodo.clear_trail()
        self._iniciar("Ejecutando el dibujo...")

    def _iniciar(self, texto):
        """Arranca la ejecución de self.plan a 50 Hz."""
        # Evita competir por target_deg/publish_target con un movimiento
        # punto a punto en curso en alguna pestaña de IK.
        self.nodo.stop_all_motion()
        self.t_inicio = time.monotonic()
        self.tick_n = 0
        self.lapiz_abajo_antes = False
        self.estado(texto, 'blue')
        self._tick()

    def _tick(self):
        inicio_tick = time.monotonic()
        plan = self.plan
        t = inicio_tick - self.t_inicio
        fin = t >= plan.duracion
        t = min(t, plan.duracion)
        ang = plan.en(plan.t[0] + t)
        self.nodo.target_deg = [float(v) for v in ang]
        self.nodo.publish_target()
        if self.var_hw.get():
            self.nodo.apply_to_hardware(lambda *_: None)

        # Rastro del pie en RViz: solo con el lápiz abajo, cortando entre trazos
        k = int(round(t / plan.dt))
        k = min(k, len(plan.t) - 1)
        if plan.lapiz_abajo[k]:
            if not self.lapiz_abajo_antes and self.nodo.trail_points:
                self.nodo.trail_cortes.append(len(self.nodo.trail_points))   # nuevo trazo
            fk = forward_kinematics_right if self.side == 'right' else forward_kinematics_left
            x, y, z = get_position(fk(self.nodo.target_deg)[-1])
            self.nodo.trail_points.append(Point(x=x / 1000.0, y=y / 1000.0, z=z / 1000.0))
        self.lapiz_abajo_antes = bool(plan.lapiz_abajo[k])

        # Lo costoso (rastro, sliders, cursor de la gráfica) se refresca
        # menos a menudo que la consigna, para sostener los 50 Hz.
        self.tick_n += 1
        if fin or self.tick_n % 10 == 0:
            self.nodo.publish_trail()
            self._sincronizar_gui(t, fin)

        if fin:
            self.after_id = None
            self.estado(f"✓ Movimiento completo ({plan.duracion:.2f} s).", 'green')
            return
        # Se descuenta lo que tardó este ciclo para mantener ~50 Hz. Igual
        # la posición se calcula con el reloj real, así que un retraso no
        # deforma el perfil: solo se envían menos muestras.
        espera_ms = int(max(1.0, (DT - (time.monotonic() - inicio_tick)) * 1000))
        self.after_id = self.nodo.root.after(espera_ms, self._tick)

    def _sincronizar_gui(self, t, fin):
        nodo = self.nodo
        # Tk avisa del cambio del slider cuando queda libre (no al hacer
        # set), así que la guarda se quita también "cuando quede libre",
        # después de ese aviso. Si no, el slider volvería a publicar el
        # ángulo redondeado a su resolución.
        nodo._sincronizando = True
        for i, v in enumerate(nodo.target_deg):
            nodo.sliders[i].set(v)
            nodo.slider_labels[i].config(text=f"{v:.1f}°")
            nodo.entries[i].delete(0, tk.END)
            nodo.entries[i].insert(0, f"{v:.1f}")
        nodo.root.after_idle(lambda: setattr(nodo, '_sincronizando', False))
        nodo.update_kinematics()
        if self.fig is not None and self.cursor and (fin or self.tick_n % 25 == 0):
            for c in self.cursor:
                c.set_xdata([self.plan.t[0] + t] * 2)
            self.canvas.draw_idle()
        if not fin:
            self.estado(f"Ejecutando... {t:5.1f} / {self.plan.duracion:.1f} s", 'blue')

    def parar(self, silencioso=False):
        if self.after_id is not None:
            try:
                self.nodo.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
            if not silencioso:
                self.estado("Movimiento detenido.", 'dark orange')
