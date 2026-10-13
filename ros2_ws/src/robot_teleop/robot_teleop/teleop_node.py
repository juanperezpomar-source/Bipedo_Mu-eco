# ============================================================
# TELEOP NODE - MUÑECO BÍPEDO
# ============================================================
#
# Este nodo proporciona una interfaz gráfica en Tkinter para:
#
#   1. Controlar las 3 articulaciones de una pierna.
#   2. Introducir manualmente los ángulos articulares.
#   3. Publicar los comandos mediante ROS 2.
#   4. Recibir el estado actual de las articulaciones.
#   5. Calcular la cinemática directa.
#   6. Mostrar las matrices homogéneas T01, T02, T03 y T04.
#   7. Mostrar la posición X, Y, Z del extremo de la cadena.
#   8. Cinemática inversa (5 métodos) y generación de
#      trayectorias (pestaña "Trayectorias").
#
# Es la interfaz de la pata del parcial adaptada al muñeco:
# cadena 0A4 (L1..L5, sin articulación fantasma) y UNA sola
# ESP32 que mueve directo los 3 servos. El botón "Aplicar a
# motores" publica /servo_commands (ángulo + offset de cada
# servo) que la ESP32 recibe por micro-ROS
# (firmware/muneco_motores). La posición real la miden los
# encoders AS5600 y llega ya traducida por el nodo
# servo_encoders (/robot/encoder_joint_states y
# /robot/encoder_status).
#
# La misma GUI se utiliza para ambas piernas.
#
# El parámetro:
#
#       leg_side = "right"
#       leg_side = "left"
#
# determina qué cadena cinemática utilizar y qué nombre
# mostrar en la interfaz.
#
# ============================================================


# ------------------------------------------------------------
# IMPORTACIONES
# ------------------------------------------------------------

import math
import os
import threading
import time

import rclpy
from rclpy.node import Node

from robot_interfaces.msg import (
    RobotCommand, JointState, IKResult, IKJacobTarget
)
from std_msgs.msg import Float32MultiArray, Float64MultiArray, String
from sensor_msgs.msg import JointState as SensorJointState
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

import tkinter as tk
from tkinter import ttk


# ------------------------------------------------------------
# IMPORTACIÓN DE CINEMÁTICA
# ------------------------------------------------------------

from robot_kinematics.cinematica_directa_der_MUNECO_izq import (
    forward_kinematics_right,
    forward_kinematics_left,
    get_position,
    L1, L2, L3, L4, L5,
)

from robot_kinematics.kinem_invers_leg_mth_MUNECO_izq import verificar_con_mth

from robot_kinematics.kinem_muneco import (
    R_DH_A_BASE_LINK,
    OFFSET_BASE_LINK_MM,
    PAPEL_RECOMENDADO,
    punto_a_base_link,
)

from robot_kinematics import perfiles_temporales as pt

from robot_teleop.trayectorias_ui import PestanaTrayectorias


# ------------------------------------------------------------
# POSICIÓN DE "HOME" PARA LA PESTAÑA DE CINEMÁTICA INVERSA (mm)
# ------------------------------------------------------------

# Pierna colgando (q = 0): el pie queda a L2+L3+L4+L5 hacia abajo y a
# -L1 en z (pierna izquierda; la derecha es el espejo en z).

HOME_X = L2 + L3 + L4 + L5
HOME_Y = 0.0
HOME_Z_LEFT = -L1


# ============================================================
# TRAYECTORIA DESDE ARCHIVO (pestaña "Trayectorias" / trazador)
# ============================================================
#
# El archivo trayectoria.txt vive en
# src/robot_kinematics/trayectorias/ y se instala con colcon en
# share/robot_kinematics/trayectorias/. Después de editarlo hay
# que volver a correr `colcon build`.
#
# Formato: una línea por punto "x, y, z" en mm (mismo sistema
# de coordenadas que los campos X, Y, Z de las pestañas de
# cinemática). Se aceptan comas o espacios como separador (y
# corchetes o paréntesis alrededor de cada punto); las líneas
# vacías y lo que va después de '#' se ignoran.
#
# ============================================================

TRAJECTORY_FILE_NAME = 'trayectoria.txt'


def _trajectory_file_path():

    from ament_index_python.packages import get_package_share_directory

    return os.path.join(
        get_package_share_directory('robot_kinematics'),
        'trayectorias',
        TRAJECTORY_FILE_NAME
    )


def _load_trajectory_file(path):

    # Devuelve una lista de (n_linea, x, y, z). Lanza ValueError
    # con el número de línea si alguna no tiene el formato.

    points = []

    with open(path, 'r', encoding='utf-8') as f:

        for line_number, line in enumerate(f, start=1):

            line = line.split('#', 1)[0].strip()

            if not line:
                continue

            # Se aceptan también puntos entre corchetes o
            # paréntesis: "[x, y, z]" o "(x, y, z)".
            for char in '[](),':
                line = line.replace(char, ' ')

            values = line.split()

            if len(values) != 3:
                raise ValueError(
                    f"línea {line_number}: se esperaban 3 valores "
                    f"(x, y, z), hay {len(values)}."
                )

            try:
                x, y, z = (float(v) for v in values)
            except ValueError:
                raise ValueError(
                    f"línea {line_number}: valor no numérico."
                )

            points.append((line_number, x, y, z))

    return points


# ============================================================
# TIPO DE DESPLAZAMIENTO (pestañas de cinemática inversa)
#
# Movimiento articular de la pose actual al punto calculado, con
# la ley temporal elegida (perfiles_temporales.py). No se incluyen
# los métodos de "puntos intermedios": aquí siempre hay un solo
# tramo, de la pose actual al destino.
#
# Único parámetro editable: T (tiempo del movimiento). v0, vf, a0,
# af NO se piden en la interfaz: quedan en 0, el caso reposo-reposo
# (empieza y termina sin velocidad ni aceleración), que es el que
# corresponde a un movimiento punto a punto como este.
# ============================================================

DT_MOVIMIENTO = 0.02   # 50 Hz, igual que la pestaña "Trayectorias"

METODOS_DESPLAZAMIENTO = (
    'lineal', 'cubico', 'quintico'
)


# ============================================================
# CLASE PRINCIPAL
# ============================================================

class TeleopNode(Node):

    def __init__(self):

        # ----------------------------------------------------
        # Inicialización del nodo ROS 2
        # ----------------------------------------------------

        super().__init__('teleop_node')


        # ----------------------------------------------------
        # PARÁMETROS ROS 2
        # ----------------------------------------------------

        self.declare_parameter(
            'leg_side',
            'right'
        )

        self.declare_parameter(
            'num_joints',
            3
        )

        # Offset de cada servo [°]: la ESP32 recibe en
        # /servo_commands el ángulo de la interfaz + offset y lo
        # resta de nuevo (ver offset en firmware/muneco_motores).
        # Deben coincidir con los del firmware.

        self.declare_parameter(
            'servo_offset_deg',
            [
                180.0,    # Hip Roll   (DS3245)
                135.0,    # Hip Pitch  (DS3240)
                135.0     # Knee Pitch (DS3240)
            ]
        )

        self.leg_side = (
            self.get_parameter('leg_side')
            .get_parameter_value()
            .string_value
            .lower()
        )

        self.num_joints = (
            self.get_parameter('num_joints')
            .get_parameter_value()
            .integer_value
        )


        # ----------------------------------------------------
        # VALIDACIÓN DE LA PIERNA
        # ----------------------------------------------------

        if self.leg_side not in ['right', 'left']:

            self.get_logger().warn(
                f"leg_side='{self.leg_side}' no válido. "
                f"Se utilizará 'right'."
            )

            self.leg_side = 'right'


        # Nombre utilizado visualmente en la interfaz.

        if self.leg_side == 'right':
            self.leg_name = 'DERECHA'
        else:
            self.leg_name = 'IZQUIERDA'


        # Z de "home" según la pierna (espejo en z).

        self.home_z = (
            -HOME_Z_LEFT if self.leg_side == 'right' else HOME_Z_LEFT
        )


        # ====================================================
        # CONFIGURACIÓN DE ARTICULACIONES
        # ====================================================

        self.joint_names = [
            'Hip Roll',
            'Hip Pitch',
            'Knee Pitch'
        ]


        # ----------------------------------------------------
        # LÍMITES VISUALES / DE ENTRADA
        # ----------------------------------------------------

        self.lower_deg = [
            0,    # Hip Roll
            -90.0,    # Hip Pitch
            -90.0     # Knee Pitch
        ]

        self.upper_deg = [
            90.0,     # Hip Roll
            90.0,     # Hip Pitch
            90.0      # Knee Pitch
        ]


        # ----------------------------------------------------
        # OFFSETS DE LOS SERVOS (parámetro servo_offset_deg)
        # ----------------------------------------------------

        self.servo_offset_deg = [
            float(v) for v in
            self.get_parameter('servo_offset_deg').value
        ]


        # ====================================================
        # VALORES ACTUALES Y OBJETIVO
        # ====================================================

        self.target_deg = [
            0.0,
            0.0,
            0.0
        ]

        self.current_deg = [
            0.0,
            0.0,
            0.0
        ]


        # ----------------------------------------------------
        # ÁNGULO REAL DE CADA ARTICULACIÓN, medido por los
        # encoders y publicado por el nodo servo_encoders en
        # /robot/encoder_joint_states (rad, sin offsets). Se
        # pasa a grados para comparar "comandado" vs "real" de
        # forma directa con los sliders.
        # ----------------------------------------------------

        self.real_deg = [
            0.0,
            0.0,
            0.0
        ]

        self.has_real_state = False
        # Nunca representar lecturas no recibidas como cero mecánico.
        self.real_valid = [False] * self.num_joints
        self.real_updated_at = [None] * self.num_joints
        self.real_timeout_sec = 2.0  # Solo seguridad de comunicación / semilla IK
        self.real_sim_tolerance_deg = 2.0  # Umbral visual encoder vs RViz
        # Estado de cada encoder según servo_encoders:
        # 'ok', 'sin_respuesta', 'sin_esp' o None (nodo sin publicar).
        self.encoder_status = [None] * self.num_joints
        self.encoder_status_at = None
        self.has_sim_state = False
        # Las cinco pestañas comparten estados medidos y simulados.
        self.ik_result_panels = {}
        # Fuente de la semilla para los dos solvers iterativos.
        # Se crean las StringVar después de inicializar tk.Tk().
        self.ik_seed_sources = {}


        # ====================================================
        # VARIABLES DE CINEMÁTICA
        # ====================================================

        self.transforms = []

        # MTH seleccionada actualmente.

        self.selected_mth = 'T04'


        # ====================================================
        # PUBLICADORES ROS 2
        # ====================================================

        # Publicador principal de comandos del robot.

        self.command_pub = self.create_publisher(
            RobotCommand,
            '/robot/command',
            10
        )


        # Publicador para mandar la pose a los motores reales:
        # va directo a la ESP32 (micro-ROS), en grados, con el
        # offset de cada servo ya sumado (data[i] = q_i + offset_i).
        # Solo se publica al apretar "Aplicar a motores" (o con
        # "+ motores" en la pestaña Trayectorias).

        self.servo_pub = self.create_publisher(
            Float32MultiArray,
            '/servo_commands',
            10
        )


        # ====================================================
        # SUSCRIPTOR DE ESTADO
        # ====================================================

        self.state_sub = self.create_subscription(
            JointState,
            '/robot/joint_states',
            self.on_state,
            10
        )


        # Posición medida por los encoders y su estado, ya
        # traducidos por el nodo servo_encoders.

        self.encoder_state_sub = self.create_subscription(
            SensorJointState,
            '/robot/encoder_joint_states',
            self.on_encoder_state,
            10
        )

        self.encoder_status_sub = self.create_subscription(
            String,
            '/robot/encoder_status',
            self.on_encoder_status,
            10
        )

        # ====================================================
        # CINEMÁTICA INVERSA (vía ik_node, por tópicos)
        # ====================================================

        self.ik_target_pub = self.create_publisher(
            Point,
            '/robot/ik_target',
            10
        )

        self.ik_target_pub_branch = self.create_publisher(
            Float64MultiArray, '/robot/ik_target_branch', 10
        )

        self.ik_result_sub = self.create_subscription(
            IKResult,
            '/robot/ik_result',
            self.on_ik_result,
            10
        )

        # Resultado pendiente de aplicar en la GUI. Se llena en
        # on_ik_result (hilo de ROS) y se consume en refresh()
        # (hilo de Tkinter), igual que current_deg/real_deg.

        self.pending_ik_result = None


        # ----------------------------------------------------
        # Movimiento punto a punto animado, uno por pestaña de IK
        # ('alg', 'newton', 'des', 'geom', 'grad'). Cada entrada
        # guarda sus widgets de tipo de trayectoria (los llena
        # build_motion_type_frame) y, mientras se está moviendo,
        # las muestras (Q, t) y el id del root.after() pendiente.
        # ----------------------------------------------------

        self.ik_motion = {}


        # ----------------------------------------------------
        # RASTRO DEL RECORRIDO (verificación visual en RViz)
        #
        # Cada vez que un objetivo de cinemática inversa es
        # alcanzable y se aplica (en cualquiera de las 5
        # pestañas de IK), se agrega el punto a esta lista y se
        # publica como un Marker tipo LINE_STRIP, en el mismo
        # frame que usa la cinemática (Base_link), para poder
        # ver en RViz el camino real que fue recorriendo el pie.
        # ----------------------------------------------------

        self.trail_points = []

        # Índices de trail_points donde empieza un trazo nuevo (lápiz
        # levantado entre medio): ahí no se dibuja la unión.
        self.trail_cortes = []

        # True mientras la pestaña de trayectorias mueve los sliders
        # por código, para que no vuelvan a publicar.
        self._sincronizando = False

        self.trail_pub = self.create_publisher(
            Marker,
            '/robot/ik_trail',
            10
        )

        # El papel recomendado (kinem_muneco.PAPEL_RECOMENDADO) se
        # dibuja en RViz como una hoja semitransparente, en el mismo
        # tópico (namespace 'papel'). Se republica cada segundo para
        # que aparezca aunque RViz se abra después.

        self.paper_timer = self.create_timer(
            1.0,
            self.publish_paper
        )


        # ====================================================
        # CINEMÁTICA INVERSA - MÉTODO DE NEWTON-RAPHSON
        # (vía ik_newton_node, por tópicos)
        # ====================================================

        self.ik_newton_target_pub = self.create_publisher(
            IKJacobTarget,
            '/robot/ik_newton_target',
            10
        )

        self.ik_newton_result_sub = self.create_subscription(
            IKResult,
            '/robot/ik_newton_result',
            self.on_ik_newton_result,
            10
        )

        self.pending_ik_newton_result = None


        # ====================================================
        # CINEMÁTICA INVERSA - MÉTODO DE DESACOPLE
        # (vía ik_des_node, por tópicos)
        # ====================================================

        self.ik_des_target_pub = self.create_publisher(
            Point,
            '/robot/ik_des_target',
            10
        )

        self.ik_des_target_pub_branch = self.create_publisher(
            Float64MultiArray, '/robot/ik_des_target_branch', 10
        )

        self.ik_des_result_sub = self.create_subscription(
            IKResult,
            '/robot/ik_des_result',
            self.on_ik_des_result,
            10
        )

        self.pending_ik_des_result = None


        # ====================================================
        # CINEMÁTICA INVERSA - MÉTODO GEOMÉTRICO
        # (vía ik_geom_node, por tópicos)
        # ====================================================

        self.ik_geom_target_pub = self.create_publisher(
            Point,
            '/robot/ik_geom_target',
            10
        )

        self.ik_geom_target_pub_branch = self.create_publisher(
            Float64MultiArray, '/robot/ik_geom_target_branch', 10
        )

        self.ik_geom_result_sub = self.create_subscription(
            IKResult,
            '/robot/ik_geom_result',
            self.on_ik_geom_result,
            10
        )

        self.pending_ik_geom_result = None


        # ====================================================
        # CINEMÁTICA INVERSA - GRADIENTE DESCENDENTE
        # (vía ik_grad_node, por tópicos; usa semilla igual
        # que Newton)
        # ====================================================

        self.ik_grad_target_pub = self.create_publisher(
            IKJacobTarget,
            '/robot/ik_grad_target',
            10
        )

        self.ik_grad_result_sub = self.create_subscription(
            IKResult,
            '/robot/ik_grad_result',
            self.on_ik_grad_result,
            10
        )

        self.pending_ik_grad_result = None


        # ====================================================
        # VERIFICACIÓN MTH (matriz de transformación homogénea)
        #
        # No usa tópicos: es un cálculo cerrado (sin iteraciones)
        # que se llama directamente desde la GUI, igual que ya se
        # hace con la cinemática directa de cinematica_directa_der_izq. Guarda
        # el último objetivo (x, y, z) pedido en cada pestaña de
        # IK para poder comparar contra la posición que realmente
        # produce la pose calculada.
        # ====================================================

        self.last_ik_target = None
        self.last_ik_newton_target = None
        self.last_ik_des_target = None
        self.last_ik_geom_target = None
        self.last_ik_grad_target = None


        # ====================================================
        # CREACIÓN DE LA INTERFAZ
        # ====================================================

        self.root = tk.Tk()

        self.root.title(
            f"Teleop muñeco bípedo - Pierna {self.leg_name}"
        )


        # ----------------------------------------------------
        # Tamaño inicial de la ventana
        # ----------------------------------------------------

        self.root.geometry("980x810")

        self.root.resizable(
            False,
            True
        )


        # ====================================================
        # ESTILO DE TKINTER
        # ====================================================

        style = ttk.Style()

        try:
            style.theme_use('clam')
        except tk.TclError:
            pass


        style.configure(
            'Title.TLabel',
            font=('Arial', 18, 'bold')
        )

        style.configure(
            'Subtitle.TLabel',
            font=('Arial', 11)
        )

        style.configure(
            'Method.TLabel',
            font=('Arial', 14, 'bold'),
            foreground='#1f4e79'
        )

        style.configure(
            'Section.TLabelframe.Label',
            font=('Arial', 11, 'bold')
        )


        # ====================================================
        # CONTENEDOR PRINCIPAL
        # ====================================================

        main_frame = ttk.Frame(
            self.root,
            padding=8
        )

        main_frame.pack(
            fill='both',
            expand=True
        )


        # ====================================================
        # TÍTULO
        # ====================================================

        title_label = ttk.Label(
            main_frame,
            text=f"TELEOPERACIÓN PIERNA {self.leg_name}",
            style='Title.TLabel'
        )

        title_label.pack(
            pady=(0, 2)
        )


        subtitle_label = ttk.Label(
            main_frame,
            text="Control de articulaciones, cinemática directa e inversa",
            style='Subtitle.TLabel'
        )

        subtitle_label.pack(
            pady=(0, 8)
        )


        # ====================================================
        # PESTAÑAS: CINEMÁTICA DIRECTA / CINEMÁTICA INVERSA
        # ====================================================

        notebook = ttk.Notebook(
            main_frame
        )

        notebook.pack(
            fill='both',
            expand=True
        )

        tab_fk = ttk.Frame(
            notebook,
            padding=0
        )

        tab_ik = ttk.Frame(
            notebook,
            padding=8
        )

        tab_ik_newton = ttk.Frame(
            notebook,
            padding=8
        )

        tab_ik_des = ttk.Frame(
            notebook,
            padding=8
        )

        tab_ik_geom = ttk.Frame(
            notebook,
            padding=8
        )

        tab_ik_grad = ttk.Frame(
            notebook,
            padding=8
        )

        # Orden solicitado: CD, Geométrico, Algebraico, MTH, Newton, Gradiente, Trayectorias.
        for pagina, etiqueta in (
            (tab_fk, "Cinemática directa"),
            (tab_ik_geom, "CI Geométrico"),
            (tab_ik, "CI Algebraico"),
            (tab_ik_des, "CI MTH Desacople"),
            (tab_ik_newton, "CI Newton"),
            (tab_ik_grad, "CI Gradiente Descendente"),
        ):
            notebook.add(pagina, text=etiqueta)

        # ----------------------------------------------------
        # El título de la ventana muestra el método de la
        # pestaña activa, para diferenciarlas mejor.
        # ----------------------------------------------------

        self.tab_titles = {
            str(tab_fk): "Cinemática directa",
            str(tab_ik): "Cinemática inversa - Método algebraico",
            str(tab_ik_newton): "Cinemática inversa - Método de Newton-Raphson",
            str(tab_ik_des): "Cinemática inversa - Método por desacople",
            str(tab_ik_geom): "Cinemática inversa - Método geométrico",
            str(tab_ik_grad): (
                "Cinemática inversa - Método de gradiente descendente"
            ),
        }

        tab_tray = ttk.Frame(
            notebook,
            padding=6
        )

        notebook.add(
            tab_tray,
            text="Trayectorias"
        )

        self.tab_titles[str(tab_tray)] = "Trayectorias"
        self.tab_tray = tab_tray

        # ----------------------------------------------------
        # Pestaña de generación de trayectorias (perfiles
        # temporales y ejecución a 50 Hz). Necesita más ancho
        # para las gráficas: la ventana se agranda mientras
        # está seleccionada (ver on_tab_changed).
        # ----------------------------------------------------

        try:
            archivo_inicial = _trajectory_file_path()
        except Exception:
            archivo_inicial = ''

        self.pestana_trayectorias = PestanaTrayectorias(
            tab_tray, self, archivo_inicial
        )

        notebook.bind(
            '<<NotebookTabChanged>>',
            lambda event: self.on_tab_changed(event.widget)
        )


        # ====================================================
        # SECCIÓN DE ARTICULACIONES
        # ====================================================

        joints_frame = ttk.LabelFrame(
            tab_fk,
            text=f"Articulaciones - Pierna {self.leg_name}",
            padding=8,
            style='Section.TLabelframe'
        )

        joints_frame.pack(
            fill='x',
            pady=(0, 8)
        )


        # Listas donde almacenaremos los elementos gráficos.

        self.sliders = []
        self.slider_labels = []
        self.real_labels = []


        # ----------------------------------------------------
        # CREACIÓN DE LOS 3 SLIDERS
        # ----------------------------------------------------

        for i in range(self.num_joints):

            name_label = ttk.Label(
                joints_frame,
                text=self.joint_names[i],
                width=12
            )

            name_label.grid(
                row=i,
                column=0,
                padx=(8, 5),
                pady=5,
                sticky='w'
            )


            # ------------------------------------------------
            # SLIDER
            # ------------------------------------------------

            slider = tk.Scale(
                joints_frame,
                from_=self.lower_deg[i],
                to=self.upper_deg[i],
                orient='horizontal',
                resolution=0.1,
                showvalue=False,
                length=350,
                command=lambda value, index=i:
                    self.on_slider_change(
                        index,
                        value
                    )
            )

            slider.set(
                self.target_deg[i]
            )

            slider.grid(
                row=i,
                column=1,
                padx=5,
                pady=2
            )


            self.sliders.append(
                slider
            )


            # ------------------------------------------------
            # ETIQUETA DEL VALOR DEL SLIDER
            # ------------------------------------------------

            value_label = ttk.Label(
                joints_frame,
                text=f"{self.target_deg[i]:.1f}°",
                width=7
            )

            value_label.grid(
                row=i,
                column=2,
                padx=(5, 8),
                pady=5
            )


            self.slider_labels.append(
                value_label
            )


            # ------------------------------------------------
            # ETIQUETA DEL ÁNGULO REAL (feedback del servo)
            # ------------------------------------------------

            real_label = ttk.Label(
                joints_frame,
                text="real: sin datos",
                width=16,
                foreground='gray'
            )

            real_label.grid(
                row=i,
                column=3,
                padx=(5, 8),
                pady=5
            )

            self.real_labels.append(
                real_label
            )


        # ====================================================
        # INGRESO MANUAL
        # ====================================================

        manual_frame = ttk.LabelFrame(
            tab_fk,
            text="Ingresar ángulos manualmente (grados)",
            padding=8,
            style='Section.TLabelframe'
        )

        manual_frame.pack(
            fill='x',
            pady=(0, 8)
        )


        # ----------------------------------------------------
        # CAMPOS DE ENTRADA
        # ----------------------------------------------------

        self.entries = []

        for i in range(self.num_joints):

            label = ttk.Label(
                manual_frame,
                text=self.joint_names[i]
            )

            label.grid(
                row=0,
                column=i * 2,
                padx=(5, 3),
                pady=3
            )


            entry = tk.Entry(
                manual_frame,
                width=8,
                justify='center'
            )

            entry.insert(
                0,
                f"{self.target_deg[i]:.1f}"
            )

            entry.grid(
                row=0,
                column=i * 2 + 1,
                padx=(0, 8),
                pady=3
            )


            self.entries.append(
                entry
            )


        # ====================================================
        # BOTÓN APLICAR
        # ====================================================

        apply_button = ttk.Button(
            manual_frame,
            text="Aplicar ángulos",
            command=self.apply_entry_angles
        )

        apply_button.grid(
            row=1,
            column=0,
            columnspan=6,
            pady=(8, 4)
        )


        # ====================================================
        # BOTÓN APLICAR A HARDWARE (ESP32 real)
        # ====================================================

        hardware_button = tk.Button(
            manual_frame,
            text="Aplicar a motores (HARDWARE)",
            bg='#ffdddd',
            command=self.apply_to_hardware
        )

        hardware_button.grid(
            row=2,
            column=0,
            columnspan=6,
            pady=(4, 4)
        )


        # ====================================================
        # MENSAJE DE ESTADO
        # ====================================================

        self.status_label = tk.Label(
            manual_frame,
            text="",
            font=('Arial', 10, 'bold'),
            anchor='center'
        )

        self.status_label.grid(
            row=3,
            column=0,
            columnspan=6,
            pady=(2, 0)
        )


        # ====================================================
        # CINEMÁTICA DIRECTA
        # ====================================================

        kinematics_frame = ttk.LabelFrame(
            tab_fk,
            text="Cinemática directa",
            padding=8,
            style='Section.TLabelframe'
        )

        kinematics_frame.pack(
            fill='x',
            pady=(0, 0)
        )


        # ====================================================
        # SELECTOR DE MTH
        # ====================================================

        mth_label = ttk.Label(
            kinematics_frame,
            text="Seleccionar MTH:"
        )

        mth_label.grid(
            row=0,
            column=0,
            padx=(5, 5),
            pady=3,
            sticky='w'
        )


        self.mth_selector = ttk.Combobox(
            kinematics_frame,
            values=[
                'T01',
                'T02',
                'T03',
                'T04'
            ],
            state='readonly',
            width=10
        )

        self.mth_selector.set(
            self.selected_mth
        )

        self.mth_selector.grid(
            row=0,
            column=1,
            padx=5,
            pady=3,
            sticky='w'
        )


        # Cada vez que cambia la MTH seleccionada,
        # actualizamos la matriz y la posición.

        self.mth_selector.bind(
            '<<ComboboxSelected>>',
            self.on_mth_selected
        )


        # ====================================================
        # MATRIZ HOMOGÉNEA
        # ====================================================

        matrix_frame = ttk.Frame(
            kinematics_frame
        )

        matrix_frame.grid(
            row=1,
            column=0,
            columnspan=2,
            pady=(8, 5)
        )


        self.matrix_labels = []


        # ----------------------------------------------------
        # Creamos una matriz visual de 4x4.
        # ----------------------------------------------------

        for row in range(4):

            matrix_row = []

            for col in range(4):

                label = tk.Label(
                    matrix_frame,
                    text="0.0000",
                    width=12,
                    relief='ridge',
                    borderwidth=1,
                    anchor='center'
                )

                label.grid(
                    row=row,
                    column=col,
                    padx=1,
                    pady=1
                )

                matrix_row.append(
                    label
                )

            self.matrix_labels.append(
                matrix_row
            )


        # ====================================================
        # POSICIÓN
        # ====================================================

        position_frame = ttk.LabelFrame(
            kinematics_frame,
            text="Posición",
            padding=8,
            style='Section.TLabelframe'
        )

        position_frame.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky='ew',
            pady=(8, 0)
        )


        # ----------------------------------------------------
        # Etiquetas X, Y y Z
        # ----------------------------------------------------

        self.position_labels = {}


        self.position_labels['x'] = ttk.Label(
            position_frame,
            text="x = 0.0000 m"
        )

        self.position_labels['x'].grid(
            row=0,
            column=0,
            padx=(20, 30),
            pady=2
        )


        self.position_labels['y'] = ttk.Label(
            position_frame,
            text="y = 0.0000 m"
        )

        self.position_labels['y'].grid(
            row=0,
            column=1,
            padx=30,
            pady=2
        )


        self.position_labels['z'] = ttk.Label(
            position_frame,
            text="z = 0.0000 m"
        )

        self.position_labels['z'].grid(
            row=0,
            column=2,
            padx=(30, 20),
            pady=2
        )


        # ====================================================
        # PESTAÑA DE CINEMÁTICA INVERSA
        # ====================================================

        self.build_method_header(
            tab_ik,
            self.tab_titles[str(tab_ik)]
        )

        ik_coords_frame = ttk.LabelFrame(
            tab_ik,
            text="Coordenada objetivo (mm)",
            padding=8,
            style='Section.TLabelframe'
        )

        ik_coords_frame.pack(
            fill='x',
            pady=(0, 8)
        )

        self.ik_entries = {}

        for i, axis in enumerate(('X', 'Y', 'Z')):

            label = ttk.Label(
                ik_coords_frame,
                text=f"{axis}:"
            )

            label.grid(
                row=0,
                column=i * 2,
                padx=(5, 3),
                pady=5
            )

            entry = tk.Entry(
                ik_coords_frame,
                width=8,
                justify='center'
            )

            entry.insert(
                0,
                "0.0"
            )

            entry.grid(
                row=0,
                column=i * 2 + 1,
                padx=(0, 10),
                pady=5
            )

            self.ik_entries[axis] = entry

        ik_send_button = ttk.Button(
            ik_coords_frame,
            text="Calcular y enviar",
            command=self.send_ik_target
        )

        ik_send_button.grid(
            row=1,
            column=0,
            columnspan=3,
            pady=(5, 0)
        )

        ik_home_button = tk.Button(
            ik_coords_frame,
            text="Home",
            bg='#ffdddd',
            command=self.go_home
        )

        ik_home_button.grid(
            row=1,
            column=3,
            columnspan=3,
            pady=(5, 0)
        )


        # ----------------------------------------------------
        # RESULTADO (q1, q2, q3)
        # ----------------------------------------------------

        # Rama cinemática usada únicamente al solicitar la solución analítica.
        self._crear_selector_codo(tab_ik, 'alg')

        ik_result_frame = ttk.LabelFrame(
            tab_ik,
            text="Resultado",
            padding=8,
            style='Section.TLabelframe'
        )

        ik_result_frame.pack(
            fill='x',
            pady=(0, 8)
        )

        # Una misma tabla en cada método: medición, simulación y objetivo.
        self.ik_q_labels = self._build_ik_result_table(
            ik_result_frame, 'alg'
        )


        # ----------------------------------------------------
        # APLICAR A HARDWARE (ESP32 real)
        #
        # Envía a los motores (/servo_commands) la última consigna calculada
        # (self.target_deg, ya sincronizada con el resultado de
        # esta pestaña). La previsualización en RViz ya ocurre
        # sola al calcular, por /robot/command.
        # ----------------------------------------------------

        ik_hardware_button = tk.Button(
            tab_ik,
            text="Aplicar a motores (HARDWARE)",
            bg='#ffdddd',
            command=lambda: self._aplicar_movimiento_hardware('alg', self.set_ik_status)
        )

        ik_hardware_button.pack(
            pady=(0, 8)
        )


        # ----------------------------------------------------
        # TIPO DE DESPLAZAMIENTO (ley temporal del movimiento)
        # ----------------------------------------------------

        self.build_motion_type_frame(
            tab_ik,
            'alg'
        )


        # ----------------------------------------------------
        # VERIFICACIÓN MTH DEL RESULTADO
        # ----------------------------------------------------

        self.ik_mth = self.build_mth_verification(tab_ik)


        # ----------------------------------------------------
        # MENSAJE DE ESTADO (pestaña IK)
        # ----------------------------------------------------

        self.ik_status_label = tk.Label(
            tab_ik,
            text="Ingrese una coordenada y presione \"Calcular y enviar\".",
            font=('Arial', 10, 'bold'),
            wraplength=380,
            justify='center'
        )

        self.ik_status_label.pack(
            pady=(5, 0)
        )


        # ====================================================
        # PESTAÑA DE CINEMÁTICA INVERSA (NEWTON)
        # ====================================================

        self.build_method_header(
            tab_ik_newton,
            self.tab_titles[str(tab_ik_newton)]
        )

        ikn_coords_frame = ttk.LabelFrame(
            tab_ik_newton,
            text="Coordenada objetivo (mm)",
            padding=8,
            style='Section.TLabelframe'
        )

        ikn_coords_frame.pack(
            fill='x',
            pady=(0, 8)
        )

        self.ikn_entries = {}

        for i, axis in enumerate(('X', 'Y', 'Z')):

            label = ttk.Label(
                ikn_coords_frame,
                text=f"{axis}:"
            )

            label.grid(
                row=0,
                column=i * 2,
                padx=(5, 3),
                pady=5
            )

            entry = tk.Entry(
                ikn_coords_frame,
                width=8,
                justify='center'
            )

            entry.insert(
                0,
                "0.0"
            )

            entry.grid(
                row=0,
                column=i * 2 + 1,
                padx=(0, 10),
                pady=5
            )

            self.ikn_entries[axis] = entry

        ikn_send_button = ttk.Button(
            ikn_coords_frame,
            text="Calcular y enviar",
            command=self.send_ik_newton_target
        )

        ikn_send_button.grid(
            row=1,
            column=0,
            columnspan=3,
            pady=(5, 0)
        )

        ikn_home_button = tk.Button(
            ikn_coords_frame,
            text="Home",
            bg='#ffdddd',
            command=self.go_home_newton
        )

        ikn_home_button.grid(
            row=1,
            column=3,
            columnspan=3,
            pady=(5, 0)
        )


        # ----------------------------------------------------
        # RESULTADO (q1, q2, q3)
        # ----------------------------------------------------

        # Dos opciones excluyentes para la semilla de IK.
        # Ninguna selección activa motores ni crea una conexión serial.
        self._build_ik_seed_selector(tab_ik_newton, 'newton')

        ikn_result_frame = ttk.LabelFrame(
            tab_ik_newton,
            text="Resultado",
            padding=8,
            style='Section.TLabelframe'
        )

        ikn_result_frame.pack(
            fill='x',
            pady=(0, 8)
        )

        # Una misma tabla en cada método: medición, simulación y objetivo.
        self.ikn_q_labels = self._build_ik_result_table(
            ikn_result_frame, 'newton'
        )


        # ----------------------------------------------------
        # APLICAR A HARDWARE (ESP32 real)
        # ----------------------------------------------------

        ikn_hardware_button = tk.Button(
            tab_ik_newton,
            text="Aplicar a motores (HARDWARE)",
            bg='#ffdddd',
            command=lambda: self._aplicar_movimiento_hardware('newton', self.set_ik_newton_status)
        )

        ikn_hardware_button.pack(
            pady=(0, 8)
        )


        # ----------------------------------------------------
        # TIPO DE DESPLAZAMIENTO (ley temporal del movimiento)
        # ----------------------------------------------------

        self.build_motion_type_frame(
            tab_ik_newton,
            'newton'
        )


        # ----------------------------------------------------
        # VERIFICACIÓN MTH DEL RESULTADO
        # ----------------------------------------------------

        self.ikn_mth = self.build_mth_verification(tab_ik_newton)


        # ----------------------------------------------------
        # MENSAJE DE ESTADO (pestaña Newton)
        # ----------------------------------------------------

        self.ikn_status_label = tk.Label(
            tab_ik_newton,
            text="Ingrese objetivo y semilla, luego \"Calcular y enviar\".",
            font=('Arial', 10, 'bold'),
            wraplength=380,
            justify='center'
        )

        self.ikn_status_label.pack(
            pady=(5, 0)
        )


        # ====================================================
        # PESTAÑA DE CINEMÁTICA INVERSA (DESACOPLE)
        # ====================================================

        self.build_method_header(
            tab_ik_des,
            self.tab_titles[str(tab_ik_des)]
        )

        ikd_coords_frame = ttk.LabelFrame(
            tab_ik_des,
            text="Coordenada objetivo (mm)",
            padding=8,
            style='Section.TLabelframe'
        )

        ikd_coords_frame.pack(
            fill='x',
            pady=(0, 8)
        )

        self.ikd_entries = {}

        for i, axis in enumerate(('X', 'Y', 'Z')):

            label = ttk.Label(
                ikd_coords_frame,
                text=f"{axis}:"
            )

            label.grid(
                row=0,
                column=i * 2,
                padx=(5, 3),
                pady=5
            )

            entry = tk.Entry(
                ikd_coords_frame,
                width=8,
                justify='center'
            )

            entry.insert(
                0,
                "0.0"
            )

            entry.grid(
                row=0,
                column=i * 2 + 1,
                padx=(0, 10),
                pady=5
            )

            self.ikd_entries[axis] = entry

        ikd_send_button = ttk.Button(
            ikd_coords_frame,
            text="Calcular y enviar",
            command=self.send_ik_des_target
        )

        ikd_send_button.grid(
            row=1,
            column=0,
            columnspan=3,
            pady=(5, 0)
        )

        ikd_home_button = tk.Button(
            ikd_coords_frame,
            text="Home",
            bg='#ffdddd',
            command=self.go_home_des
        )

        ikd_home_button.grid(
            row=1,
            column=3,
            columnspan=3,
            pady=(5, 0)
        )


        # ----------------------------------------------------
        # RESULTADO (q1, q2, q3)
        # ----------------------------------------------------

        # Rama cinemática usada únicamente al solicitar la solución analítica.
        self._crear_selector_codo(tab_ik_des, 'des')

        ikd_result_frame = ttk.LabelFrame(
            tab_ik_des,
            text="Resultado",
            padding=8,
            style='Section.TLabelframe'
        )

        ikd_result_frame.pack(
            fill='x',
            pady=(0, 8)
        )

        # Una misma tabla en cada método: medición, simulación y objetivo.
        self.ikd_q_labels = self._build_ik_result_table(
            ikd_result_frame, 'des'
        )


        # ----------------------------------------------------
        # APLICAR A HARDWARE (ESP32 real)
        # ----------------------------------------------------

        ikd_hardware_button = tk.Button(
            tab_ik_des,
            text="Aplicar a motores (HARDWARE)",
            bg='#ffdddd',
            command=lambda: self._aplicar_movimiento_hardware('des', self.set_ik_des_status)
        )

        ikd_hardware_button.pack(
            pady=(0, 8)
        )


        # ----------------------------------------------------
        # TIPO DE DESPLAZAMIENTO (ley temporal del movimiento)
        # ----------------------------------------------------

        self.build_motion_type_frame(
            tab_ik_des,
            'des'
        )


        # ----------------------------------------------------
        # VERIFICACIÓN MTH DEL RESULTADO
        # ----------------------------------------------------

        self.ikd_mth = self.build_mth_verification(tab_ik_des)


        # ----------------------------------------------------
        # MENSAJE DE ESTADO (pestaña Desacople)
        # ----------------------------------------------------

        self.ikd_status_label = tk.Label(
            tab_ik_des,
            text="Ingrese una coordenada y presione \"Calcular y enviar\".",
            font=('Arial', 10, 'bold'),
            wraplength=380,
            justify='center'
        )

        self.ikd_status_label.pack(
            pady=(5, 0)
        )


        # ====================================================
        # PESTAÑA DE CINEMÁTICA INVERSA (GEOMÉTRICO)
        # ====================================================

        self.build_method_header(
            tab_ik_geom,
            self.tab_titles[str(tab_ik_geom)]
        )

        ikg_coords_frame = ttk.LabelFrame(
            tab_ik_geom,
            text="Coordenada objetivo (mm)",
            padding=8,
            style='Section.TLabelframe'
        )

        ikg_coords_frame.pack(
            fill='x',
            pady=(0, 8)
        )

        self.ikg_entries = {}

        for i, axis in enumerate(('X', 'Y', 'Z')):

            label = ttk.Label(
                ikg_coords_frame,
                text=f"{axis}:"
            )

            label.grid(
                row=0,
                column=i * 2,
                padx=(5, 3),
                pady=5
            )

            entry = tk.Entry(
                ikg_coords_frame,
                width=8,
                justify='center'
            )

            entry.insert(
                0,
                "0.0"
            )

            entry.grid(
                row=0,
                column=i * 2 + 1,
                padx=(0, 10),
                pady=5
            )

            self.ikg_entries[axis] = entry

        ikg_send_button = ttk.Button(
            ikg_coords_frame,
            text="Calcular y enviar",
            command=self.send_ik_geom_target
        )

        ikg_send_button.grid(
            row=1,
            column=0,
            columnspan=3,
            pady=(5, 0)
        )

        ikg_home_button = tk.Button(
            ikg_coords_frame,
            text="Home",
            bg='#ffdddd',
            command=self.go_home_geom
        )

        ikg_home_button.grid(
            row=1,
            column=3,
            columnspan=3,
            pady=(5, 0)
        )


        # ----------------------------------------------------
        # RESULTADO (q1, q2, q3)
        # ----------------------------------------------------

        # Rama cinemática usada únicamente al solicitar la solución analítica.
        self._crear_selector_codo(tab_ik_geom, 'geom')

        ikg_result_frame = ttk.LabelFrame(
            tab_ik_geom,
            text="Resultado",
            padding=8,
            style='Section.TLabelframe'
        )

        ikg_result_frame.pack(
            fill='x',
            pady=(0, 8)
        )

        # Una misma tabla en cada método: medición, simulación y objetivo.
        self.ikg_q_labels = self._build_ik_result_table(
            ikg_result_frame, 'geom'
        )


        # ----------------------------------------------------
        # APLICAR A HARDWARE (ESP32 real)
        # ----------------------------------------------------

        ikg_hardware_button = tk.Button(
            tab_ik_geom,
            text="Aplicar a motores (HARDWARE)",
            bg='#ffdddd',
            command=lambda: self._aplicar_movimiento_hardware('geom', self.set_ik_geom_status)
        )

        ikg_hardware_button.pack(
            pady=(0, 8)
        )


        # ----------------------------------------------------
        # TIPO DE DESPLAZAMIENTO (ley temporal del movimiento)
        # ----------------------------------------------------

        self.build_motion_type_frame(
            tab_ik_geom,
            'geom'
        )


        # ----------------------------------------------------
        # VERIFICACIÓN MTH DEL RESULTADO
        # ----------------------------------------------------

        self.ikg_mth = self.build_mth_verification(tab_ik_geom)


        # ----------------------------------------------------
        # MENSAJE DE ESTADO (pestaña Geométrico)
        # ----------------------------------------------------

        self.ikg_status_label = tk.Label(
            tab_ik_geom,
            text="Ingrese una coordenada y presione \"Calcular y enviar\".",
            font=('Arial', 10, 'bold'),
            wraplength=380,
            justify='center'
        )

        self.ikg_status_label.pack(
            pady=(5, 0)
        )


        # ====================================================
        # PESTAÑA DE CINEMÁTICA INVERSA (GRADIENTE DESCENDENTE)
        # ====================================================

        self.build_method_header(
            tab_ik_grad,
            self.tab_titles[str(tab_ik_grad)]
        )

        ikgr_coords_frame = ttk.LabelFrame(
            tab_ik_grad,
            text="Coordenada objetivo (mm)",
            padding=8,
            style='Section.TLabelframe'
        )

        ikgr_coords_frame.pack(
            fill='x',
            pady=(0, 8)
        )

        self.ikgr_entries = {}

        for i, axis in enumerate(('X', 'Y', 'Z')):

            label = ttk.Label(
                ikgr_coords_frame,
                text=f"{axis}:"
            )

            label.grid(
                row=0,
                column=i * 2,
                padx=(5, 3),
                pady=5
            )

            entry = tk.Entry(
                ikgr_coords_frame,
                width=8,
                justify='center'
            )

            entry.insert(
                0,
                "0.0"
            )

            entry.grid(
                row=0,
                column=i * 2 + 1,
                padx=(0, 10),
                pady=5
            )

            self.ikgr_entries[axis] = entry

        ikgr_send_button = ttk.Button(
            ikgr_coords_frame,
            text="Calcular y enviar",
            command=self.send_ik_grad_target
        )

        ikgr_send_button.grid(
            row=1,
            column=0,
            columnspan=3,
            pady=(5, 0)
        )

        ikgr_home_button = tk.Button(
            ikgr_coords_frame,
            text="Home",
            bg='#ffdddd',
            command=self.go_home_grad
        )

        ikgr_home_button.grid(
            row=1,
            column=3,
            columnspan=3,
            pady=(5, 0)
        )


        # ----------------------------------------------------
        # RESULTADO (q1, q2, q3)
        # ----------------------------------------------------

        # Dos opciones excluyentes para la semilla de IK.
        # Ninguna selección activa motores ni crea una conexión serial.
        self._build_ik_seed_selector(tab_ik_grad, 'grad')

        ikgr_result_frame = ttk.LabelFrame(
            tab_ik_grad,
            text="Resultado",
            padding=8,
            style='Section.TLabelframe'
        )

        ikgr_result_frame.pack(
            fill='x',
            pady=(0, 8)
        )

        # Una misma tabla en cada método: medición, simulación y objetivo.
        self.ikgr_q_labels = self._build_ik_result_table(
            ikgr_result_frame, 'grad'
        )


        # ----------------------------------------------------
        # APLICAR A HARDWARE (ESP32 real)
        # ----------------------------------------------------

        ikgr_hardware_button = tk.Button(
            tab_ik_grad,
            text="Aplicar a motores (HARDWARE)",
            bg='#ffdddd',
            command=lambda: self._aplicar_movimiento_hardware('grad', self.set_ik_grad_status)
        )

        ikgr_hardware_button.pack(
            pady=(0, 8)
        )


        # ----------------------------------------------------
        # TIPO DE DESPLAZAMIENTO (ley temporal del movimiento)
        # ----------------------------------------------------

        self.build_motion_type_frame(
            tab_ik_grad,
            'grad'
        )


        # ----------------------------------------------------
        # VERIFICACIÓN MTH DEL RESULTADO
        # ----------------------------------------------------

        self.ikgr_mth = self.build_mth_verification(tab_ik_grad)


        # ----------------------------------------------------
        # MENSAJE DE ESTADO (pestaña Gradiente descendente)
        # ----------------------------------------------------

        self.ikgr_status_label = tk.Label(
            tab_ik_grad,
            text="Ingrese una coordenada y presione \"Calcular y enviar\".",
            font=('Arial', 10, 'bold'),
            wraplength=380,
            justify='center'
        )

        self.ikgr_status_label.pack(
            pady=(5, 0)
        )


        # ====================================================
        # ACTUALIZACIÓN INICIAL DE CINEMÁTICA
        # ====================================================

        self.update_kinematics()


        # ====================================================
        # ACTUALIZACIÓN PERIÓDICA DE LA GUI
        # ====================================================

        self.root.after(
            50,
            self.refresh
        )


        # ====================================================
        # CIERRE DE LA VENTANA
        # ====================================================

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.on_close
        )


    # ========================================================
    # SLIDER
    # ========================================================

    def _build_ik_result_table(self, frame, key):
        """Construye la misma tabla en las cinco pestañas de IK.

        Los objetivos solo se escriben tras el cálculo del solver;
        RViz procede del estado simulado; Robot real del feedback.
        """
        frame.columnconfigure(0, weight=2)
        for col in (1, 2, 3):
            frame.columnconfigure(col, weight=1)

        for col, title in enumerate(
            ('Articulación', 'Robot real (ESP32)',
             'RViz (simulación)', 'Objetivo IK'),
        ):
            ttk.Label(frame, text=title, font=('Arial', 9, 'bold')).grid(
                row=0, column=col, sticky='w', padx=7, pady=(3, 8)
            )

        panels = {'real': [], 'sim': [], 'target': []}
        names = ('q1 (Hip Roll)', 'q2 (Hip Pitch)', 'q3 (Knee)')
        for i, name in enumerate(names):
            ttk.Label(frame, text=name + ':').grid(
                row=i + 1, column=0, sticky='w', padx=7, pady=4
            )
            for col, kind in enumerate(('real', 'sim', 'target'), start=1):
                label = ttk.Label(frame, text='Sin datos' if kind == 'real' else '--')
                label.grid(row=i + 1, column=col, sticky='w', padx=7, pady=4)
                panels[kind].append(label)
        self.ik_result_panels[key] = panels
        # Compatibilidad: apply_ik_*_result usa las etiquetas del objetivo.
        return panels['target']

    def _build_ik_seed_selector(self, parent, method):
        """Dos botones de opción para la semilla del solver iterativo."""
        frame = ttk.LabelFrame(
            parent, text="Semilla inicial para el cálculo IK",
            padding=8, style='Section.TLabelframe'
        )
        frame.pack(fill='x', pady=(0, 8))
        selected = tk.StringVar(master=self.root, value='rviz')
        self.ik_seed_sources[method] = selected
        ttk.Radiobutton(
            frame, text="Simulación (posición actual de RViz)",
            variable=selected, value='rviz'
        ).pack(side='left', padx=(4, 16))
        ttk.Radiobutton(
            frame, text="Hardware (posición que reporta la ESP32)",
            variable=selected, value='hardware'
        ).pack(side='left', padx=4)
        ttk.Label(
            parent,
            text="Hardware requiere 3 lecturas válidas y recientes de los encoders (máx. 2 s).",
            foreground='#666666'
        ).pack(anchor='w', pady=(0, 7))

    def _get_ik_seed(self, method):
        """Devuelve q inicial en GRADOS en el orden Roll, Pitch, Knee.

        No reutiliza estados obsoletos ni sustituye hardware por RViz
        silenciosamente. Los callbacks ROS actualizan los estados existentes.
        """
        source = self.ik_seed_sources[method].get()
        if source == 'rviz':
            if not self.has_sim_state:
                raise ValueError(
                    "No hay estado articular de RViz. Espere /robot/joint_states "
                    "o ajuste la configuración simulada antes de calcular."
                )
            seed = list(self.current_deg[:3])
        elif source == 'hardware':
            now = time.monotonic()
            names = ('Hip Roll', 'Hip Pitch', 'Knee')
            missing = [
                names[i] for i in range(3)
                if (not self.real_valid[i]
                    or self.real_updated_at[i] is None
                    or (now - self.real_updated_at[i]) > self.real_timeout_sec)
            ]
            if missing:
                raise ValueError(
                    "No es posible usar semilla de hardware: sin lectura "
                    "válida y reciente de " + ', '.join(missing) + ". "
                    "Conecte la ESP32 (micro_ros_agent) o seleccione RViz."
                )
            seed = list(self.real_deg[:3])
        else:
            raise ValueError("Fuente de semilla desconocida.")
        if len(seed) != 3 or not all(math.isfinite(v) for v in seed):
            raise ValueError("La semilla contiene valores articulares inválidos.")
        return seed, source

    def _real_joint_visual_status(self, i, now=None):
        """Separar disponibilidad de medida y coincidencia con RViz.

        Un valor cercano a RViz no prueba que el sensor siga conectado;
        una lectura reciente distinta de RViz no es una lectura antigua.
        """
        if now is None:
            now = time.monotonic()
        if (self.encoder_status_at is not None
                and now - self.encoder_status_at <= self.real_timeout_sec):
            if self.encoder_status[i] == 'sin_respuesta':
                return 'encoder_falla'
            if self.encoder_status[i] == 'sin_esp':
                return 'sin_esp'
        stamp = self.real_updated_at[i]
        if not self.real_valid[i] or stamp is None:
            return 'sin_datos'
        if now - stamp > self.real_timeout_sec:
            return 'sin_actualizacion'
        if not self.has_sim_state:
            return 'sin_simulacion'
        if abs(self.real_deg[i] - self.current_deg[i]) > self.real_sim_tolerance_deg:
            return 'desfasado'
        return 'coincide'

    def _refresh_ik_measurement_tables(self):
        """Actualizar UI desde Tkinter, nunca desde callbacks ROS."""
        now = time.monotonic()
        for panels in self.ik_result_panels.values():
            for i in range(min(3, self.num_joints)):
                real = panels['real'][i]
                state = self._real_joint_visual_status(i, now)
                if state == 'sin_datos':
                    real.config(text='Sin datos', foreground='gray')
                elif state == 'encoder_falla':
                    real.config(text='Encoder sin respuesta', foreground='#b22222')
                elif state == 'sin_esp':
                    real.config(text='ESP32 sin conexión', foreground='gray')
                else:
                    value = f'{self.real_deg[i]:.2f}°'
                    if state == 'sin_actualizacion':
                        real.config(text=value + ' (sin actualización)', foreground='gray')
                    elif state == 'sin_simulacion':
                        real.config(text=value + ' (sin RViz)', foreground='#a56c09')
                    elif state == 'desfasado':
                        real.config(text=value + ' (difiere >2°)', foreground='#b22222')
                    else:
                        real.config(text=value, foreground='#14683a')

                sim = panels['sim'][i]
                if self.has_sim_state:
                    sim.config(text=f'{self.current_deg[i]:.2f}°')
                else:
                    sim.config(text='--')

    def _update_real_joint(self, index, value):
        if 0 <= index < self.num_joints and math.isfinite(value):
            self.real_deg[index] = float(value)
            self.real_valid[index] = True
            self.real_updated_at[index] = time.monotonic()
            self.has_real_state = any(self.real_valid)


    def on_slider_change(
        self,
        index,
        value
    ):

        if self._sincronizando:
            return

        try:
            value = float(value)

        except ValueError:
            return


        # Guardamos el nuevo valor.

        self.target_deg[index] = value


        # Actualizamos el texto del slider.

        self.slider_labels[index].config(
            text=f"{value:.1f}°"
        )


        # Actualizamos el campo de entrada manual.

        self.entries[index].delete(
            0,
            tk.END
        )

        self.entries[index].insert(
            0,
            f"{value:.1f}"
        )


        # Publicamos el comando.

        self.publish_target()


        # Recalculamos la cinemática.

        self.update_kinematics()


    # ========================================================
    # APLICAR ÁNGULOS MANUALES
    # ========================================================

    def apply_entry_angles(self):

        # ----------------------------------------------------
        # 1. Intentamos convertir todos los campos a float.
        # ----------------------------------------------------

        try:

            values = [
                float(
                    self.entries[i].get()
                )
                for i in range(
                    self.num_joints
                )
            ]

        except ValueError:

            self.set_status(
                "Error: todos los ángulos deben ser valores numéricos.",
                "red"
            )

            return


        # ----------------------------------------------------
        # 2. VALIDAMOS LOS LÍMITES
        # ----------------------------------------------------

        for i, value in enumerate(values):

            if (
                value < self.lower_deg[i]
                or
                value > self.upper_deg[i]
            ):

                self.set_status(
                    f"{self.joint_names[i]} debe estar entre "
                    f"{self.lower_deg[i]:.1f}° y "
                    f"{self.upper_deg[i]:.1f}°.",
                    "red"
                )

                return


        # ----------------------------------------------------
        # 3. TODOS LOS VALORES SON VÁLIDOS
        # ----------------------------------------------------

        self.target_deg = values.copy()


        # ----------------------------------------------------
        # Actualizamos sliders y etiquetas.
        # ----------------------------------------------------

        for i, value in enumerate(values):

            self.sliders[i].set(
                value
            )

            self.slider_labels[i].config(
                text=f"{value:.1f}°"
            )


        # ----------------------------------------------------
        # 4. Publicamos el nuevo comando.
        # ----------------------------------------------------

        self.publish_target()


        # ----------------------------------------------------
        # 5. Recalculamos la cinemática.
        # ----------------------------------------------------

        self.update_kinematics()


        # ----------------------------------------------------
        # 6. Mensaje de éxito.
        # ----------------------------------------------------

        self.set_status(
            "✓ Ángulos aplicados correctamente.",
            "green"
        )


    # ========================================================
    # MENSAJE DE ESTADO
    # ========================================================

    def set_status(
        self,
        message,
        color
    ):

        self.status_label.config(
            text=message,
            foreground=color
        )


    # ========================================================
    # PUBLICAR COMANDO
    # ========================================================

    def publish_target(self):

        # ----------------------------------------------------
        # Conversión grados -> radianes
        # ----------------------------------------------------

        angles_rad = [
            math.radians(angle)
            for angle in self.target_deg
        ]


        # ----------------------------------------------------
        # Mensaje RobotCommand (preview, siempre en vivo).
        #
        # OJO: esto ya NO toca los motores reales. Solo mueve
        # el modelo en RViz. Para mandar la orden al hardware
        # hay que presionar el botón "Aplicar a motores".
        # ----------------------------------------------------

        command_msg = RobotCommand()

        command_msg.position = angles_rad

        self.command_pub.publish(
            command_msg
        )


    # ========================================================
    # APLICAR A HARDWARE (ESP32 real, directo por /servo_commands)
    # ========================================================

    def apply_to_hardware(self, set_status=None, on_rejected=None, goal_deg=None):

        # Envía al hardware real la última consigna calculada,
        # sea por los sliders (Cinemática directa) o por el
        # resultado de cualquiera de las pestañas de cinemática
        # inversa: todas actualizan self.target_deg al llegar.
        #
        # set_status permite avisar en la pestaña desde la que se
        # apretó el botón (si no se pasa, se usa la de Cinemática
        # directa, por compatibilidad con el botón original).
        #
        # on_rejected (opcional) se llama si el comando no se
        # envía (la usa la trayectoria para detenerse).
        #
        # Devuelve False si la interfaz lo rechazó sin enviarlo.

        if set_status is None:
            set_status = self.set_status

        angles = (list(self.target_deg) if goal_deg is None
                  else list(goal_deg))
        if len(angles) != self.num_joints or not all(
            math.isfinite(v) for v in angles
        ):
            set_status('✗ Consigna articular inválida.', 'red')
            return False

        if len(self.servo_offset_deg) != self.num_joints or not all(
            math.isfinite(v) for v in self.servo_offset_deg
        ):
            set_status('✗ Parámetro servo_offset_deg inválido.', 'red')
            if on_rejected is not None:
                on_rejected()
            return False


        # ----------------------------------------------------
        # 1. LÍMITE DE LA INTERFAZ (lower_deg / upper_deg)
        #
        # Única barrera antes de la ESP32 (en el muñeco no pasa
        # por control_node): si la consigna se sale del rango de
        # la interfaz (puede pasar con resultados de IK), no se
        # envía nada. La ESP32 además recorta al recorrido
        # físico de cada servo.
        # ----------------------------------------------------

        tol = 1e-6

        fuera = [
            f"{self.joint_names[i]}={angles[i]:.1f}° "
            f"({self.lower_deg[i]:.0f}° a {self.upper_deg[i]:.0f}°)"
            for i in range(self.num_joints)
            if not (
                self.lower_deg[i] - tol
                <= angles[i]
                <= self.upper_deg[i] + tol
            )
        ]

        if fuera:

            set_status(
                "✗ No se envió a los motores, fuera de rango: "
                + ", ".join(fuera),
                "red"
            )

            if on_rejected is not None:
                on_rejected()

            return False


        # ----------------------------------------------------
        # 2. ENVIAR A LA ESP32 (grados + offset de cada servo)
        # ----------------------------------------------------

        servo_msg = Float32MultiArray()

        servo_msg.data = [
            float(angles[i] + self.servo_offset_deg[i])
            for i in range(self.num_joints)
        ]

        self.servo_pub.publish(
            servo_msg
        )

        texto = ', '.join(f'{v:.1f}' for v in angles)

        if self.servo_pub.get_subscription_count() == 0:

            set_status(
                f"⚠ Publicado [{texto}]°, pero nadie escucha "
                "/servo_commands (¿micro_ros_agent y ESP32 conectados?).",
                "dark orange"
            )

        else:

            set_status(
                f"✓ Enviado a la ESP32: [{texto}]°",
                "green"
            )

        return True


    # ========================================================
    # CINEMÁTICA INVERSA - ENVIAR OBJETIVO
    # ========================================================

    def _crear_selector_codo(self, parent, metodo):
        """Selector independiente para cada uno de los tres métodos analíticos."""
        if not hasattr(self, 'ik_codo_vars'):
            self.ik_codo_vars = {}
        var = tk.StringVar(master=self.root, value='arriba')
        self.ik_codo_vars[metodo] = var
        frame = ttk.LabelFrame(parent, text='Configuración de rodilla', padding=8)
        frame.pack(fill='x', pady=(0, 8))
        ttk.Radiobutton(
            frame, text='Codo arriba', variable=var, value='arriba'
        ).pack(side='left', padx=(5, 20))
        ttk.Radiobutton(
            frame, text='Codo abajo', variable=var, value='abajo'
        ).pack(side='left')

    def _publicar_ik_con_rama(self, publicador, x, y, z, metodo):
        """Float64MultiArray: [x_mm, y_mm, z_mm, signo_de_rama]."""
        seleccion = self.ik_codo_vars[metodo].get()
        signo = 1.0 if seleccion == 'arriba' else -1.0
        msg = Float64MultiArray()
        msg.data = [float(x), float(y), float(z), signo]
        publicador.publish(msg)

    def send_ik_target(self):

        try:
            x = float(self.ik_entries['X'].get())
            y = float(self.ik_entries['Y'].get())
            z = float(self.ik_entries['Z'].get())

        except ValueError:

            self.set_ik_status(
                "Error: X, Y, Z deben ser valores numéricos.",
                "red"
            )

            return

        self.last_ik_target = (x, y, z)

        self._publicar_ik_con_rama(self.ik_target_pub_branch, x, y, z, 'alg')

        self.set_ik_status(
            "Calculando...",
            "gray"
        )


    # ========================================================
    # CINEMÁTICA INVERSA - IR A HOME
    # ========================================================

    def go_home(self):

        self.ik_entries['X'].delete(0, tk.END)
        self.ik_entries['X'].insert(0, f"{HOME_X}")

        self.ik_entries['Y'].delete(0, tk.END)
        self.ik_entries['Y'].insert(0, f"{HOME_Y}")

        self.ik_entries['Z'].delete(0, tk.END)
        self.ik_entries['Z'].insert(0, f"{self.home_z}")

        self.send_ik_target()


    # ========================================================
    # TIPO DE DESPLAZAMIENTO - WIDGETS DE UNA PESTAÑA
    #
    # Se llama una vez por cada pestaña de cinemática inversa.
    # Construye el selector de ley temporal (lineal / cúbico /
    # quíntico) y sus campos de parámetros, para el movimiento
    # articular de la pose actual hasta el punto calculado
    # (ver _mover_ik).
    # ========================================================

    def build_motion_type_frame(self, parent, key):

        motion_frame = ttk.LabelFrame(
            parent,
            text="Tipo de desplazamiento",
            padding=8,
            style='Section.TLabelframe'
        )

        motion_frame.pack(
            fill='x',
            pady=(0, 8)
        )

        metodo_var = tk.StringVar(value=pt.NOMBRES['quintico'])

        metodo_combo = ttk.Combobox(
            motion_frame,
            textvariable=metodo_var,
            state='readonly',
            width=40,
            values=[pt.NOMBRES[m] for m in METODOS_DESPLAZAMIENTO]
        )

        metodo_combo.grid(
            row=0,
            column=0,
            columnspan=4,
            sticky='w',
            padx=(5, 3),
            pady=(5, 3)
        )

        campos = {}

        def campo(clave, texto, fila, col):

            ttk.Label(
                motion_frame,
                text=texto
            ).grid(
                row=fila,
                column=col,
                sticky='w',
                padx=(5, 2),
                pady=2
            )

            entry = tk.Entry(
                motion_frame,
                width=7,
                justify='center'
            )

            entry.grid(
                row=fila,
                column=col + 1,
                sticky='w',
                padx=(0, 8),
                pady=2
            )

            campos[clave] = entry

        campo('T', "T (s):", 1, 0)

        # T por defecto: sin él, el movimiento falla con "falta el
        # tiempo T" y la pierna no se mueve aunque la IK calcule q.
        # v0, vf, a0, af no son editables: siempre reposo-reposo (0).
        campos['T'].insert(0, "2")

        aviso_label = ttk.Label(
            motion_frame,
            text="",
            foreground='#555',
            wraplength=330,
            justify='left'
        )

        aviso_label.grid(
            row=5,
            column=0,
            columnspan=4,
            sticky='w',
            padx=(5, 3),
            pady=(3, 0)
        )

        coord_var = tk.BooleanVar(value=True)

        # Coordinacion isocrona siempre activa (sin control visible).
        self.ik_motion[key] = {
            'metodo_var': metodo_var,
            'campos': campos,
            'coord_var': coord_var,
            'aviso_label': aviso_label,
            'after_id': None,
        }

        metodo_combo.bind(
            '<<ComboboxSelected>>',
            lambda event: self._actualizar_campos_desplazamiento(key)
        )

        self._actualizar_campos_desplazamiento(key)


    # ========================================================
    # TIPO DE DESPLAZAMIENTO - MÉTODO ELEGIDO EN UNA PESTAÑA
    # ========================================================

    def _metodo_desplazamiento(self, key):

        inverso = {texto: metodo for metodo, texto in pt.NOMBRES.items()}

        return inverso[self.ik_motion[key]['metodo_var'].get()]


    # ========================================================
    # TIPO DE DESPLAZAMIENTO - HABILITAR CAMPOS SEGÚN EL MÉTODO
    #
    # Ya no hay campos que habilitar/deshabilitar (solo queda T,
    # que siempre está activo): se deja la función porque la
    # llama el combobox al cambiar de método, por si en el futuro
    # un método necesita mostrar algo acá.
    # ========================================================

    def _actualizar_campos_desplazamiento(self, key):

        motion = self.ik_motion[key]
        motion['aviso_label'].config(text="")


    # ========================================================
    # TIPO DE DESPLAZAMIENTO - LEER MÉTODO Y PARÁMETROS
    #
    # Devuelve (metodo, T, params, coordinado). Lanza ValueError /
    # pt.ErrorPerfil si T falta o no es numérico. v0, vf, a0, af no
    # se leen de la interfaz: quedan en 0 (reposo-reposo), el
    # default de cubico()/quintico() en perfiles_temporales.py.
    # ========================================================

    def _leer_parametros_movimiento(self, key):

        motion = self.ik_motion[key]
        metodo = self._metodo_desplazamiento(key)
        campos = motion['campos']

        texto = campos['T'].get().strip()
        T = float(texto) if texto else None

        if T is None:
            raise pt.ErrorPerfil('falta el tiempo T del movimiento')

        return metodo, T, {}, True


    # ========================================================
    # DETENER CUALQUIER MOVIMIENTO ANIMADO EN CURSO
    #
    # Se llama antes de empezar uno nuevo (en cualquier pestaña
    # de IK o en la pestaña "Trayectorias"), para que no compitan
    # dos bucles por self.target_deg / publish_target a la vez.
    # ========================================================

    def stop_all_motion(self):

        for motion in self.ik_motion.values():

            if motion.get('after_id') is not None:

                try:
                    self.root.after_cancel(motion['after_id'])
                except Exception:
                    pass

                motion['after_id'] = None

        pestana = getattr(self, 'pestana_trayectorias', None)

        if pestana is not None:
            pestana.parar(silencioso=True)


    # ========================================================
    # MOVIMIENTO PUNTO A PUNTO - INICIAR
    #
    # Mueve la pierna desde la pose actual (self.target_deg)
    # hasta q_destino_deg con la ley temporal elegida en la
    # pestaña, animando a 50 Hz en vez de saltar de golpe.
    # Se llama desde cada apply_ik_*_result cuando el punto es
    # alcanzable.
    # ========================================================

    def _mover_ik(
        self, key, q_destino_deg, q_labels, mth_widgets, set_status,
        objetivo_xyz
    ):

        self.stop_all_motion()

        try:
            metodo, T, params, coordinado = self._leer_parametros_movimiento(key)

            perfiles = pt.planificar_ptp(
                list(self.target_deg), q_destino_deg, metodo, T, params,
                coordinado
            )

            _, Q, _, _ = pt.muestrear(perfiles, DT_MOVIMIENTO)

        except (pt.ErrorPerfil, ValueError, KeyError) as error:

            set_status(
                f"✗ {error or 'revisa los parámetros de la trayectoria'}",
                "red"
            )

            return

        motion = self.ik_motion[key]

        motion.update({
            'Q': Q,
            'idx': 0,
            'after_id': None,
            'q_labels': q_labels,
            'mth_widgets': mth_widgets,
            'set_status': set_status,
            'objetivo_xyz': objetivo_xyz,
            # Esto es solo preview en RViz. 'con_hardware' en False
            # explícito: si una ejecución anterior lo había dejado en
            # True, un Home/resultado de IK nuevo no debe mandar a los
            # motores sin que se apriete "Aplicar a motores" de nuevo.
            'con_hardware': False,
        })

        set_status(
            "Moviendo...",
            "blue"
        )

        self._tick_ik_motion(key)


    # ========================================================
    # MOVIMIENTO PUNTO A PUNTO - UN PASO (50 Hz)
    # ========================================================

    def _tick_ik_motion(self, key):

        motion = self.ik_motion[key]
        Q = motion['Q']
        idx = motion['idx']

        q = [float(v) for v in Q[idx]]
        self.target_deg = q

        # Mientras se actualizan sliders/entradas por código, no deben
        # volver a publicar (igual que en trayectorias_ui._sincronizar_gui).
        self._sincronizando = True

        for i in range(min(self.num_joints, len(q))):

            self.sliders[i].set(q[i])

            self.slider_labels[i].config(
                text=f"{q[i]:.1f}°"
            )

            self.entries[i].delete(0, tk.END)
            self.entries[i].insert(0, f"{q[i]:.1f}")

        self.root.after_idle(lambda: setattr(self, '_sincronizando', False))

        self.publish_target()
        self.update_kinematics()

        # Si esta corrida viene de "Aplicar a motores" (ver
        # _aplicar_movimiento_hardware), además de RViz se manda cada
        # paso a /servo_commands, igual que hace la pestaña
        # "Trayectorias" con su casilla "+ motores". set_status se
        # ignora acá (se usaría para "enviando...", 50 veces por
        # segundo); el resultado final se avisa al terminar, abajo.
        if motion.get('con_hardware'):
            self.apply_to_hardware(lambda *_: None)

        # El objetivo IK es fijo durante la animación; los valores de
        # RViz los proporciona /robot/joint_states (no el objetivo).

        fin = idx >= len(Q) - 1

        if fin:

            motion['after_id'] = None

            if motion['objetivo_xyz'] is not None:

                x_obj, y_obj, z_obj = motion['objetivo_xyz']
                q_rad = [math.radians(v) for v in q]

                self.update_mth_verification(
                    motion['mth_widgets'],
                    q_rad[0], q_rad[1], q_rad[2],
                    x_obj, y_obj, z_obj
                )

                self.add_trail_point(x_obj, y_obj, z_obj)

            self.show_ik_done_status(
                motion['set_status'], q, motion.get('con_hardware', False)
            )

            return

        motion['idx'] = idx + 1

        motion['after_id'] = self.root.after(
            int(DT_MOVIMIENTO * 1000),
            lambda: self._tick_ik_motion(key)
        )


    # ========================================================
    # APLICAR A MOTORES - MISMO MOVIMIENTO QUE EN RVIZ
    #
    # A diferencia de apply_to_hardware() (salto instantáneo al
    # ángulo actual), esto repite el mismo tipo de perfil que
    # _mover_ik usa para la animación de RViz, pero publicando
    # también en /servo_commands en cada uno de los pasos
    # de 50 Hz - igual que hace la pestaña "Trayectorias" con su
    # casilla "+ motores". No usa ningún tópico nuevo.
    #
    # Se recalcula desde la pose real actual (self.target_deg)
    # hasta el mismo destino de la última animación (el último
    # punto de motion['Q']), con el método y T que estén puestos
    # AHORA en "Tipo de desplazamiento" - así, si ya se dejó
    # terminar la animación en RViz, el tramo es trivial (ya está
    # ahí); si se aprieta a mitad de camino, arranca desde ahí.
    # ========================================================

    def _aplicar_movimiento_hardware(self, key, set_status):
        """Envia el objetivo IK sin modificar la postura de RViz.

        La columna Robot real solo varía si llega telemetría auténtica.
        El firmware decide cómo ejecutar el objetivo físico (rampa de
        VEL_DEG_S en firmware/muneco_motores).
        """
        motion = self.ik_motion.get(key)
        if motion is None or motion.get('Q') is None:
            set_status('✗ Primero calcula una solución IK válida.', 'red')
            return

        goal = [float(v) for v in motion['Q'][-1]]
        if not all(math.isfinite(v) for v in goal):
            set_status('✗ Objetivo IK inválido.', 'red')
            return

        # No cambiar self.target_deg: el preview puede estar animándose.
        # El botón de hardware NO reinicia el movimiento simulado.
        self.apply_to_hardware(set_status, goal_deg=goal)


    # ========================================================
    # RASTRO DEL RECORRIDO - AGREGAR PUNTO Y PUBLICAR
    #
    # x, y, z llegan en milímetros (igual que el resto de la
    # cinemática); Marker/RViz trabajan en metros, por eso se
    # dividen entre 1000 acá.
    # ========================================================

    def add_trail_point(self, x, y, z):

        point = Point()
        point.x = x / 1000.0
        point.y = y / 1000.0
        point.z = z / 1000.0

        self.trail_points.append(point)

        self.publish_trail()


    def publish_trail(self):

        marker = Marker()

        marker.header.frame_id = 'Base_link'
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = 'ik_trail'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        # ----------------------------------------------------
        # Los puntos están en el marco base de la DH
        # (cinematica_directa_der_MUNECO_izq). En el URDF del
        # muñeco el Base_link está girado ~-90° en Z respecto a
        # esa base (X_DH -> -Y, Y_DH -> +X, Z_DH -> +Z) y
        # desplazado (kinem_muneco.R_DH_A_BASE_LINK y
        # OFFSET_BASE_LINK_MM, ajustados contra el URDF). Se
        # aplica con la pose del Marker, así los puntos quedan
        # tal cual vienen de la cinemática.
        # ----------------------------------------------------

        offset = OFFSET_BASE_LINK_MM[self.leg_side] / 1000.0

        marker.pose.position.x = float(offset[0])
        marker.pose.position.y = float(offset[1])
        marker.pose.position.z = float(offset[2])

        # Cuaternión de R_DH_A_BASE_LINK (giro de ~-90° en Z).
        R = R_DH_A_BASE_LINK[self.leg_side]
        w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0

        marker.pose.orientation.x = float((R[2, 1] - R[1, 2]) / (4.0 * w))
        marker.pose.orientation.y = float((R[0, 2] - R[2, 0]) / (4.0 * w))
        marker.pose.orientation.z = float((R[1, 0] - R[0, 1]) / (4.0 * w))
        marker.pose.orientation.w = float(w)

        marker.scale.x = 0.005

        marker.color.r = 1.0
        marker.color.g = 0.1
        marker.color.b = 0.5
        marker.color.a = 1.0

        # Pares de puntos consecutivos, saltando las uniones entre
        # trazos (trail_cortes) que deja la pestaña de trayectorias.
        cortes = set(self.trail_cortes)
        puntos = []
        for i in range(1, len(self.trail_points)):
            if i not in cortes:
                puntos.append(self.trail_points[i - 1])
                puntos.append(self.trail_points[i])
        marker.points = puntos

        self.trail_pub.publish(marker)


    # ========================================================
    # RASTRO DEL RECORRIDO - LIMPIAR
    # ========================================================

    def _marker_pose_dh(self, marker, punto_mm):
        # Pose de un Marker cuyos ejes locales son los de la base
        # DH: posición = punto (mm, sistema DH) llevado a Base_link,
        # orientación = R_DH_A_BASE_LINK como cuaternión.

        p = punto_a_base_link(punto_mm, self.leg_side) / 1000.0

        marker.pose.position.x = float(p[0])
        marker.pose.position.y = float(p[1])
        marker.pose.position.z = float(p[2])

        R = R_DH_A_BASE_LINK[self.leg_side]
        w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0

        marker.pose.orientation.x = float((R[2, 1] - R[1, 2]) / (4.0 * w))
        marker.pose.orientation.y = float((R[0, 2] - R[2, 0]) / (4.0 * w))
        marker.pose.orientation.z = float((R[1, 0] - R[0, 1]) / (4.0 * w))
        marker.pose.orientation.w = float(w)


    def publish_paper(self):

        papel = PAPEL_RECOMENDADO[self.leg_side]
        y_c, z_c = papel['centro']
        espesor = 1.0   # mm, hacia abajo (+X DH) desde la superficie

        marker = Marker()

        marker.header.frame_id = 'Base_link'
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = 'papel'
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        self._marker_pose_dh(
            marker,
            (papel['x_papel'] + espesor / 2.0, y_c, z_c)
        )

        marker.scale.x = espesor / 1000.0
        marker.scale.y = papel['lado'] / 1000.0
        marker.scale.z = papel['lado'] / 1000.0

        marker.color.r = 0.95
        marker.color.g = 0.95
        marker.color.b = 0.85
        marker.color.a = 0.6

        self.trail_pub.publish(marker)


    def clear_trail(self):

        self.trail_points = []
        self.trail_cortes = []

        self.publish_trail()


    # ========================================================
    # CINEMÁTICA INVERSA - RECEPCIÓN DEL RESULTADO
    # (hilo de ROS; solo guarda el dato, ver refresh())
    # ========================================================

    def on_ik_result(self, msg):

        self.pending_ik_result = msg


    # ========================================================
    # CINEMÁTICA INVERSA - APLICAR RESULTADO
    # (hilo de Tkinter, llamado desde refresh())
    # ========================================================

    def apply_ik_result(self, result):

        if not result.reachable:

            for label in self.ik_q_labels:
                label.config(text="--", foreground='')

            self.set_ik_status(
                "✗ Posición no alcanzable.",
                "red"
            )

            self.reset_mth_verification(self.ik_mth)

            return

        q_deg = [math.degrees(q) for q in result.position]

        self.show_ik_q_labels(self.ik_q_labels, q_deg)


        # ----------------------------------------------------
        # Movemos la pierna de la pose actual al destino con la
        # ley temporal elegida en "Tipo de desplazamiento" (en
        # vez de saltar de golpe). _mover_ik anima a 50 Hz, deja
        # self.target_deg/sliders/entradas sincronizados y, al
        # terminar, verifica MTH y agrega el punto al rastro.
        # ----------------------------------------------------

        self._mover_ik(
            'alg', q_deg, self.ik_q_labels, self.ik_mth,
            self.set_ik_status, self.last_ik_target
        )


    # ========================================================
    # CINEMÁTICA INVERSA - MENSAJE DE ESTADO
    # ========================================================

    # ========================================================
    # CINEMÁTICA INVERSA - LÍMITES ARTICULARES DEL RESULTADO
    #
    # Los métodos de IK devuelven la solución exacta aunque se
    # salga de lower_deg/upper_deg (solo avisan en el log del
    # nodo). control_node la recorta para RViz, así que el
    # modelo queda en el límite y el pie NO llega al objetivo.
    # Estas funciones lo hacen visible en todas las pestañas.
    # ========================================================

    def ik_limit_violations(self, q_deg):

        tol = 1e-6

        return [
            f"q{i + 1}={q_deg[i]:.1f}° "
            f"({self.lower_deg[i]:.0f}° a {self.upper_deg[i]:.0f}°)"
            for i in range(min(self.num_joints, len(q_deg)))
            if not (
                self.lower_deg[i] - tol
                <= q_deg[i]
                <= self.upper_deg[i] + tol
            )
        ]


    def show_ik_q_labels(self, labels, q_deg):

        # Ángulo fuera de rango -> en rojo.

        for i, (label, value) in enumerate(zip(labels, q_deg)):

            fuera = not (
                self.lower_deg[i] - 1e-6
                <= value
                <= self.upper_deg[i] + 1e-6
            )

            label.config(
                text=f"{value:.2f}°" + ("  ⚠" if fuera else ""),
                foreground='red' if fuera else ''
            )


    def show_ik_done_status(self, set_status, q_deg, con_hardware=False):

        fuera = self.ik_limit_violations(q_deg)

        if fuera:

            set_status(
                "⚠ Solución fuera de los límites articulares: "
                + ", ".join(fuera)
                + ". RViz la muestra recortada al límite (el pie "
                "no llega al objetivo) y no se enviará a los motores.",
                "dark orange"
            )

        else:

            set_status(
                "✓ Movimiento realizado"
                + (" y enviado a los motores." if con_hardware else "."),
                "green"
            )


    def set_ik_status(self, message, color):

        self.ik_status_label.config(
            text=message,
            foreground=color
        )


    # ========================================================
    # CINEMÁTICA INVERSA (NEWTON) - ENVIAR OBJETIVO
    # ========================================================

    def send_ik_newton_target(self):

        try:
            x = float(self.ikn_entries['X'].get())
            y = float(self.ikn_entries['Y'].get())
            z = float(self.ikn_entries['Z'].get())

        except ValueError:

            self.set_ik_newton_status(
                "Error: X, Y, Z deben ser valores numéricos.",
                "red"
            )

            return

        self.last_ik_newton_target = (x, y, z)

        # La semilla la determina el selector de RViz / Hardware.
        # La función de validación impide usar datos físicos antiguos.

        target = IKJacobTarget()
        target.x = x
        target.y = y
        target.z = z
        try:
            seed, source = self._get_ik_seed('newton')
        except ValueError as exc:
            self.set_ik_newton_status(str(exc), "red")
            return
        target.q1_seed, target.q2_seed, target.q3_seed = seed

        self.ik_newton_target_pub.publish(target)

        self.set_ik_newton_status(
            "Calculando... semilla {}: [{:.2f}, {:.2f}, {:.2f}]°".format(
                'RViz' if source == 'rviz' else 'ESP32', *seed
            ),
            "gray"
        )


    # ========================================================
    # CINEMÁTICA INVERSA (NEWTON) - IR A HOME
    # ========================================================

    def go_home_newton(self):

        self.ikn_entries['X'].delete(0, tk.END)
        self.ikn_entries['X'].insert(0, f"{HOME_X}")

        self.ikn_entries['Y'].delete(0, tk.END)
        self.ikn_entries['Y'].insert(0, f"{HOME_Y}")

        self.ikn_entries['Z'].delete(0, tk.END)
        self.ikn_entries['Z'].insert(0, f"{self.home_z}")

        self.send_ik_newton_target()


    # ========================================================
    # CINEMÁTICA INVERSA (NEWTON) - RECEPCIÓN DEL RESULTADO
    # (hilo de ROS; solo guarda el dato, ver refresh())
    # ========================================================

    def on_ik_newton_result(self, msg):

        self.pending_ik_newton_result = msg


    # ========================================================
    # CINEMÁTICA INVERSA (NEWTON) - APLICAR RESULTADO
    # (hilo de Tkinter, llamado desde refresh())
    # ========================================================

    def apply_ik_newton_result(self, result):

        if not result.reachable:

            for label in self.ikn_q_labels:
                label.config(text="--", foreground='')

            self.set_ik_newton_status(
                "✗ No convergió / posición no alcanzable.",
                "red"
            )

            self.reset_mth_verification(self.ikn_mth)

            return

        q_deg = [math.degrees(q) for q in result.position]

        self.show_ik_q_labels(self.ikn_q_labels, q_deg)


        # ----------------------------------------------------
        # Movemos la pierna de la pose actual al destino con la
        # ley temporal elegida en "Tipo de desplazamiento" (ver
        # apply_ik_result).
        # ----------------------------------------------------

        self._mover_ik(
            'newton', q_deg, self.ikn_q_labels, self.ikn_mth,
            self.set_ik_newton_status, self.last_ik_newton_target
        )


    # ========================================================
    # CINEMÁTICA INVERSA (NEWTON) - MENSAJE DE ESTADO
    # ========================================================

    def set_ik_newton_status(self, message, color):

        self.ikn_status_label.config(
            text=message,
            foreground=color
        )


    # ========================================================
    # CINEMÁTICA INVERSA (DESACOPLE) - ENVIAR OBJETIVO
    # ========================================================

    def send_ik_des_target(self):

        try:
            x = float(self.ikd_entries['X'].get())
            y = float(self.ikd_entries['Y'].get())
            z = float(self.ikd_entries['Z'].get())

        except ValueError:

            self.set_ik_des_status(
                "Error: X, Y, Z deben ser valores numéricos.",
                "red"
            )

            return

        self.last_ik_des_target = (x, y, z)

        self._publicar_ik_con_rama(self.ik_des_target_pub_branch, x, y, z, 'des')

        self.set_ik_des_status(
            "Calculando...",
            "gray"
        )


    # ========================================================
    # CINEMÁTICA INVERSA (DESACOPLE) - IR A HOME
    # ========================================================

    def go_home_des(self):

        self.ikd_entries['X'].delete(0, tk.END)
        self.ikd_entries['X'].insert(0, f"{HOME_X}")

        self.ikd_entries['Y'].delete(0, tk.END)
        self.ikd_entries['Y'].insert(0, f"{HOME_Y}")

        self.ikd_entries['Z'].delete(0, tk.END)
        self.ikd_entries['Z'].insert(0, f"{self.home_z}")

        self.send_ik_des_target()


    # ========================================================
    # CINEMÁTICA INVERSA (DESACOPLE) - RECEPCIÓN DEL RESULTADO
    # (hilo de ROS; solo guarda el dato, ver refresh())
    # ========================================================

    def on_ik_des_result(self, msg):

        self.pending_ik_des_result = msg


    # ========================================================
    # CINEMÁTICA INVERSA (DESACOPLE) - APLICAR RESULTADO
    # (hilo de Tkinter, llamado desde refresh())
    # ========================================================

    def apply_ik_des_result(self, result):

        if not result.reachable:

            for label in self.ikd_q_labels:
                label.config(text="--", foreground='')

            self.set_ik_des_status(
                "✗ Posición no alcanzable.",
                "red"
            )

            self.reset_mth_verification(self.ikd_mth)

            return

        q_deg = [math.degrees(q) for q in result.position]

        self.show_ik_q_labels(self.ikd_q_labels, q_deg)


        # ----------------------------------------------------
        # Movemos la pierna de la pose actual al destino con la
        # ley temporal elegida en "Tipo de desplazamiento" (ver
        # apply_ik_result).
        # ----------------------------------------------------

        self._mover_ik(
            'des', q_deg, self.ikd_q_labels, self.ikd_mth,
            self.set_ik_des_status, self.last_ik_des_target
        )


    # ========================================================
    # CINEMÁTICA INVERSA (DESACOPLE) - MENSAJE DE ESTADO
    # ========================================================

    def set_ik_des_status(self, message, color):

        self.ikd_status_label.config(
            text=message,
            foreground=color
        )


    # ========================================================
    # CINEMÁTICA INVERSA (GEOMÉTRICO) - ENVIAR OBJETIVO
    # ========================================================

    def send_ik_geom_target(self):

        try:
            x = float(self.ikg_entries['X'].get())
            y = float(self.ikg_entries['Y'].get())
            z = float(self.ikg_entries['Z'].get())

        except ValueError:

            self.set_ik_geom_status(
                "Error: X, Y, Z deben ser valores numéricos.",
                "red"
            )

            return

        self.last_ik_geom_target = (x, y, z)

        self._publicar_ik_con_rama(self.ik_geom_target_pub_branch, x, y, z, 'geom')

        self.set_ik_geom_status(
            "Calculando...",
            "gray"
        )


    # ========================================================
    # CINEMÁTICA INVERSA (GEOMÉTRICO) - IR A HOME
    # ========================================================

    def go_home_geom(self):

        self.ikg_entries['X'].delete(0, tk.END)
        self.ikg_entries['X'].insert(0, f"{HOME_X}")

        self.ikg_entries['Y'].delete(0, tk.END)
        self.ikg_entries['Y'].insert(0, f"{HOME_Y}")

        self.ikg_entries['Z'].delete(0, tk.END)
        self.ikg_entries['Z'].insert(0, f"{self.home_z}")

        self.send_ik_geom_target()


    # ========================================================
    # CINEMÁTICA INVERSA (GEOMÉTRICO) - RECEPCIÓN DEL RESULTADO
    # (hilo de ROS; solo guarda el dato, ver refresh())
    # ========================================================

    def on_ik_geom_result(self, msg):

        self.pending_ik_geom_result = msg


    # ========================================================
    # CINEMÁTICA INVERSA (GEOMÉTRICO) - APLICAR RESULTADO
    # (hilo de Tkinter, llamado desde refresh())
    # ========================================================

    def apply_ik_geom_result(self, result):

        if not result.reachable:

            for label in self.ikg_q_labels:
                label.config(text="--", foreground='')

            self.set_ik_geom_status(
                "✗ Posición no alcanzable.",
                "red"
            )

            self.reset_mth_verification(self.ikg_mth)

            return

        q_deg = [math.degrees(q) for q in result.position]

        self.show_ik_q_labels(self.ikg_q_labels, q_deg)


        # ----------------------------------------------------
        # Movemos la pierna de la pose actual al destino con la
        # ley temporal elegida en "Tipo de desplazamiento" (ver
        # apply_ik_result).
        # ----------------------------------------------------

        self._mover_ik(
            'geom', q_deg, self.ikg_q_labels, self.ikg_mth,
            self.set_ik_geom_status, self.last_ik_geom_target
        )


    # ========================================================
    # CINEMÁTICA INVERSA (GEOMÉTRICO) - MENSAJE DE ESTADO
    # ========================================================

    def set_ik_geom_status(self, message, color):

        self.ikg_status_label.config(
            text=message,
            foreground=color
        )


    # ========================================================
    # CINEMÁTICA INVERSA (GRADIENTE DESCENDENTE) - ENVIAR OBJETIVO
    # ========================================================

    def send_ik_grad_target(self):

        try:
            x = float(self.ikgr_entries['X'].get())
            y = float(self.ikgr_entries['Y'].get())
            z = float(self.ikgr_entries['Z'].get())

        except ValueError:

            self.set_ik_grad_status(
                "Error: X, Y, Z deben ser valores numéricos.",
                "red"
            )

            return

        self.last_ik_grad_target = (x, y, z)

        # La semilla la determina el selector de RViz / Hardware.
        # La función de validación impide usar datos físicos antiguos.

        target = IKJacobTarget()
        target.x = x
        target.y = y
        target.z = z
        try:
            seed, source = self._get_ik_seed('grad')
        except ValueError as exc:
            self.set_ik_grad_status(str(exc), "red")
            return
        target.q1_seed, target.q2_seed, target.q3_seed = seed

        self.ik_grad_target_pub.publish(target)

        self.set_ik_grad_status(
            "Calculando... semilla {}: [{:.2f}, {:.2f}, {:.2f}]°".format(
                'RViz' if source == 'rviz' else 'ESP32', *seed
            ),
            "gray"
        )


    # ========================================================
    # CINEMÁTICA INVERSA (GRADIENTE DESCENDENTE) - IR A HOME
    # ========================================================

    def go_home_grad(self):

        self.ikgr_entries['X'].delete(0, tk.END)
        self.ikgr_entries['X'].insert(0, f"{HOME_X}")

        self.ikgr_entries['Y'].delete(0, tk.END)
        self.ikgr_entries['Y'].insert(0, f"{HOME_Y}")

        self.ikgr_entries['Z'].delete(0, tk.END)
        self.ikgr_entries['Z'].insert(0, f"{self.home_z}")

        self.send_ik_grad_target()


    # ========================================================
    # CINEMÁTICA INVERSA (GRADIENTE DESCENDENTE) - RECEPCIÓN DEL RESULTADO
    # (hilo de ROS; solo guarda el dato, ver refresh())
    # ========================================================

    def on_ik_grad_result(self, msg):

        self.pending_ik_grad_result = msg


    # ========================================================
    # CINEMÁTICA INVERSA (GRADIENTE DESCENDENTE) - APLICAR RESULTADO
    # (hilo de Tkinter, llamado desde refresh())
    # ========================================================

    def apply_ik_grad_result(self, result):

        if not result.reachable:

            for label in self.ikgr_q_labels:
                label.config(text="--", foreground='')

            self.set_ik_grad_status(
                "✗ No convergió / posición no alcanzable.",
                "red"
            )

            self.reset_mth_verification(self.ikgr_mth)

            return

        q_deg = [math.degrees(q) for q in result.position]

        self.show_ik_q_labels(self.ikgr_q_labels, q_deg)


        # ----------------------------------------------------
        # Movemos la pierna de la pose actual al destino con la
        # ley temporal elegida en "Tipo de desplazamiento" (ver
        # apply_ik_result).
        # ----------------------------------------------------

        self._mover_ik(
            'grad', q_deg, self.ikgr_q_labels, self.ikgr_mth,
            self.set_ik_grad_status, self.last_ik_grad_target
        )


    # ========================================================
    # CINEMÁTICA INVERSA (GRADIENTE DESCENDENTE) - MENSAJE DE ESTADO
    # ========================================================

    def set_ik_grad_status(self, message, color):

        self.ikgr_status_label.config(
            text=message,
            foreground=color
        )


    # ========================================================
    # ENCABEZADO CON EL NOMBRE DEL MÉTODO (pestañas de IK)
    # ========================================================

    def build_method_header(self, parent, text):

        ttk.Label(
            parent,
            text=text,
            style='Method.TLabel'
        ).pack(
            pady=(0, 6)
        )


    # ========================================================
    # CAMBIO DE PESTAÑA: título de la ventana con el método
    # ========================================================

    def on_tab_changed(self, notebook):

        method = self.tab_titles.get(notebook.select(), "")

        self.root.title(
            f"Teleop muñeco bípedo - Pierna {self.leg_name}"
            + (f" - {method}" if method else "")
        )

        # La pestaña "Trayectorias" necesita más ancho para sus
        # gráficas: la ventana se agranda mientras está seleccionada.
        if notebook.select() == str(self.tab_tray):
            self.root.geometry("1180x860")
        else:
            self.root.geometry("980x740")


    # ========================================================
    # VERIFICACIÓN MTH - CONSTRUCCIÓN DEL WIDGET
    #
    # Se llama una vez por cada pestaña de cinemática inversa
    # (algebraica, Newton, desacople, geométrico, gradiente). Devuelve un diccionario
    # con las referencias gráficas que después actualizan
    # update_mth_verification() / reset_mth_verification().
    # ========================================================

    def build_mth_verification(self, parent):

        frame = ttk.LabelFrame(
            parent,
            text="Verificación MTH (matriz de transformación homogénea)",
            padding=8,
            style='Section.TLabelframe'
        )

        frame.pack(
            fill='x',
            pady=(0, 8)
        )

        matrix_frame = ttk.Frame(frame)

        matrix_frame.pack(
            pady=(0, 5)
        )

        matrix_labels = []

        for row in range(4):

            matrix_row = []

            for col in range(4):

                label = tk.Label(
                    matrix_frame,
                    text="--",
                    width=10,
                    relief='ridge',
                    borderwidth=1,
                    anchor='center'
                )

                label.grid(
                    row=row,
                    column=col,
                    padx=1,
                    pady=1
                )

                matrix_row.append(label)

            matrix_labels.append(matrix_row)

        status_label = tk.Label(
            frame,
            text="Sin verificar todavía.",
            font=('Arial', 9, 'bold'),
            anchor='center'
        )

        status_label.pack(
            pady=(2, 0)
        )

        return {
            'matrix_labels': matrix_labels,
            'status_label': status_label,
        }


    # ========================================================
    # VERIFICACIÓN MTH - ACTUALIZAR CON UN RESULTADO
    #
    # Arma Tdes a partir del (q1,q2,q3) que calculó el método de
    # la pestaña (algebraico / Newton / desacople), le pide al
    # método MTH que recupere esos mismos ángulos leyendo la
    # matriz, y muestra la comparación.
    # ========================================================

    def update_mth_verification(
        self, mth_widgets, q1, q2, q3, x_obj, y_obj, z_obj
    ):

        resultado = verificar_con_mth(q1, q2, q3, x_obj, y_obj, z_obj)

        T = resultado['T']

        for row in range(4):
            for col in range(4):
                mth_widgets['matrix_labels'][row][col].config(
                    text=f"{T[row, col]:.4f}"
                )

        if not resultado['ok']:

            mth_widgets['status_label'].config(
                text="✗ La verificación MTH no coincide (revisar).",
                foreground='red'
            )

            return

        mth_widgets['status_label'].config(
            text=(
                f"✓ MTH verificado — error posición: "
                f"{resultado['pos_error']:.4f} mm · "
                f"error ángulos: "
                f"{math.degrees(resultado['q_error']):.5f}°"
            ),
            foreground='dark green'
        )


    # ========================================================
    # VERIFICACIÓN MTH - LIMPIAR (cuando no hubo solución)
    # ========================================================

    def reset_mth_verification(self, mth_widgets):

        for row in mth_widgets['matrix_labels']:
            for label in row:
                label.config(text="--")

        mth_widgets['status_label'].config(
            text="Sin verificar (no alcanzable).",
            foreground='gray'
        )


    # ========================================================
    # CINEMÁTICA DIRECTA
    # ========================================================

    def update_kinematics(self):

        q = self.target_deg.copy()


        # ----------------------------------------------------
        # Selección de la cadena cinemática
        # ----------------------------------------------------

        if self.leg_side == 'right':

            self.transforms = (
                forward_kinematics_right(q)
            )

        else:

            self.transforms = (
                forward_kinematics_left(q)
            )


        # ----------------------------------------------------
        # Actualizamos la representación gráfica.
        # ----------------------------------------------------

        self.update_kinematics_display()


    # ========================================================
    # ACTUALIZACIÓN DE MATRIZ Y POSICIÓN
    # ========================================================

    def update_kinematics_display(self):

        # ----------------------------------------------------
        # Si todavía no hay matrices calculadas, no hacemos
        # nada.
        # ----------------------------------------------------

        if not self.transforms:
            return


        # ----------------------------------------------------
        # Determinamos qué índice corresponde a la MTH.
        #
        # T01 -> índice 0
        # T02 -> índice 1
        # T03 -> índice 2
        # T04 -> índice 3
        # ----------------------------------------------------

        mth_index = int(
            self.selected_mth[-1]
        ) - 1


        # Verificación para evitar acceder a una posición
        # inexistente.

        if (
            mth_index < 0
            or
            mth_index >= len(self.transforms)
        ):
            return


        T = self.transforms[mth_index]


        # ====================================================
        # ACTUALIZAR MATRIZ
        # ====================================================

        for row in range(4):

            for col in range(4):

                value = T[row, col]

                self.matrix_labels[row][col].config(
                    text=f"{value:.4f}"
                )


        # ====================================================
        # EXTRAER POSICIÓN
        # ====================================================

        position = get_position(T)

        x = position[0]
        y = position[1]
        z = position[2]


        self.position_labels['x'].config(
            text=f"x = {x:.4f} m"
        )

        self.position_labels['y'].config(
            text=f"y = {y:.4f} m"
        )

        self.position_labels['z'].config(
            text=f"z = {z:.4f} m"
        )


    # ========================================================
    # CAMBIO DE MTH
    # ========================================================

    def on_mth_selected(
        self,
        event=None
    ):

        self.selected_mth = (
            self.mth_selector.get()
        )

        self.update_kinematics_display()


    # ========================================================
    # RECIBIR ESTADO DE ARTICULACIONES
    # ========================================================

    def on_state(
        self,
        msg
    ):

        try:

            positions = msg.position

            for i in range(
                min(
                    self.num_joints,
                    len(positions)
                )
            ):

                self.current_deg[i] = (
                    math.degrees(
                        positions[i]
                    )
                )
            self.has_sim_state = len(positions) >= self.num_joints

        except Exception as e:

            self.get_logger().error(
                f"Error procesando JointState: {e}"
            )


    # ========================================================
    # RECIBIR ÁNGULO REAL MEDIDO (nodo servo_encoders)
    # ========================================================

    def on_encoder_state(
        self,
        msg
    ):

        try:

            n = min(
                self.num_joints,
                len(msg.position)
            )

            for i in range(n):

                # NaN = encoder sin respuesta: _update_real_joint
                # lo descarta y la GUI muestra el estado.

                self._update_real_joint(
                    i, math.degrees(msg.position[i])
                )

        except Exception as e:

            self.get_logger().error(
                f"Error procesando /robot/encoder_joint_states: {e}"
            )

    def on_encoder_status(self, msg):

        estados = msg.data.split(',')

        for i in range(min(self.num_joints, len(estados))):
            self.encoder_status[i] = estados[i].strip()

        self.encoder_status_at = time.monotonic()


    # ========================================================
    # ACTUALIZACIÓN PERIÓDICA DE LA GUI
    # ========================================================

    def refresh(self):

        # Un error al aplicar un resultado NO debe cortar este
        # ciclo: si no se reprograma, la interfaz deja de
        # aplicar resultados de IK para siempre. Por eso el
        # trabajo va en _refresh_once() y el after() siempre
        # se ejecuta.

        try:

            self._refresh_once()

        except Exception as e:

            self.get_logger().error(
                f"Error actualizando la interfaz: {e}"
            )

        self.root.after(
            50,
            self.refresh
        )


    def _refresh_once(self):

        # ----------------------------------------------------
        # Si llegó un resultado nuevo de cinemática inversa,
        # lo aplicamos acá (hilo de Tkinter), nunca dentro de
        # on_ik_result (hilo de ROS).
        # ----------------------------------------------------

        if self.pending_ik_result is not None:

            pendiente = self.pending_ik_result
            self.pending_ik_result = None

            self.apply_ik_result(pendiente)

        if self.pending_ik_newton_result is not None:

            pendiente = self.pending_ik_newton_result
            self.pending_ik_newton_result = None

            self.apply_ik_newton_result(pendiente)

        if self.pending_ik_des_result is not None:

            pendiente = self.pending_ik_des_result
            self.pending_ik_des_result = None

            self.apply_ik_des_result(pendiente)

        if self.pending_ik_geom_result is not None:

            pendiente = self.pending_ik_geom_result
            self.pending_ik_geom_result = None

            self.apply_ik_geom_result(pendiente)

        if self.pending_ik_grad_result is not None:

            pendiente = self.pending_ik_grad_result
            self.pending_ik_grad_result = None

            self.apply_ik_grad_result(pendiente)


        # ----------------------------------------------------
        # Reflejamos en la GUI el último ángulo real recibido
        # del servo (o "sin datos" si todavía no llegó nada).
        #
        # Esto se hace acá, en el hilo de Tkinter, y no dentro
        # de on_encoder_state (que corre en el hilo de ROS), para
        # no tocar widgets desde otro hilo.
        # ----------------------------------------------------

        self._refresh_ik_measurement_tables()

        now = time.monotonic()
        for i in range(self.num_joints):
            state = self._real_joint_visual_status(i, now)
            if state == 'sin_datos':
                txt, color = 'real: sin datos', 'gray'
            elif state == 'encoder_falla':
                txt, color = 'real: encoder sin respuesta', 'firebrick'
            elif state == 'sin_esp':
                txt, color = 'real: ESP32 sin conexión', 'gray'
            else:
                value = f'real: {self.real_deg[i]:.1f}°'
                if state == 'sin_actualizacion':
                    txt, color = value + ' (sin actualización)', 'gray'
                elif state == 'sin_simulacion':
                    txt, color = value + ' (sin RViz)', '#a56c09'
                elif state == 'desfasado':
                    txt, color = value + ' (difiere >2°)', 'firebrick'
                else:
                    txt, color = value, 'dark green'
            self.real_labels[i].config(text=txt, foreground=color)



    # ========================================================
    # CIERRE
    # ========================================================

    def on_close(self):

        try:

            self.destroy_node()

        except Exception:
            pass


        try:

            if rclpy.ok():
                rclpy.shutdown()

        except Exception:
            pass


        self.root.destroy()


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main(args=None):

    # --------------------------------------------------------
    # Inicializamos ROS 2
    # --------------------------------------------------------

    rclpy.init(
        args=args
    )


    # --------------------------------------------------------
    # Creamos el nodo
    # --------------------------------------------------------

    node = TeleopNode()


    # --------------------------------------------------------
    # ROS 2 se ejecuta en un hilo separado.
    # --------------------------------------------------------

    ros_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    )

    ros_thread.start()


    # --------------------------------------------------------
    # Ejecutamos la interfaz gráfica.
    # --------------------------------------------------------

    try:

        node.root.mainloop()

    except KeyboardInterrupt:

        pass

    finally:

        # ----------------------------------------------------
        # Si la interfaz todavía está abierta, cerramos
        # correctamente ROS.
        # ----------------------------------------------------

        if rclpy.ok():

            try:
                rclpy.shutdown()

            except Exception:
                pass


        try:
            node.destroy_node()

        except Exception:
            pass


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == '__main__':
    main()
