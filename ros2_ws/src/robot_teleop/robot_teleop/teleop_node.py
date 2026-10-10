# ============================================================
# TELEOP NODE - ROBOT BÍPEDO
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
#   8. Calcular la cinemática inversa (algebraico, Newton,
#      desacople, geométrico y gradiente descendente) a través
#      de los nodos ik_*_node, y verificar cada resultado con
#      el método MTH.
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
import threading

import rclpy
from rclpy.node import Node

from robot_interfaces.msg import (
    RobotCommand, JointState, IKResult, IKJacobTarget
)
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point

import tkinter as tk
from tkinter import ttk


# ------------------------------------------------------------
# IMPORTACIÓN DE CINEMÁTICA
# ------------------------------------------------------------

from robot_kinematics.cinematica_directa_der_MUNECO_izq import (
    forward_kinematics_right,
    forward_kinematics_left,
    get_position,
    L1, L2, L3, L4, L5
)

from robot_kinematics.kinem_invers_leg_mth_MUNECO_izq import verificar_con_mth


# ------------------------------------------------------------
# POSICIÓN DE "HOME" PARA LAS PESTAÑAS DE CINEMÁTICA INVERSA
# (mm, q = 0: pierna estirada). Para la pierna derecha Z
# cambia de signo (+L1).
# ------------------------------------------------------------

HOME_X = L2 + L3 + L4 + L5
HOME_Y = 0.0
HOME_Z_LEFT = -L1


# ------------------------------------------------------------
# MÉTODOS DE CINEMÁTICA INVERSA (una pestaña por método)
#
#   topic : prefijo de los tópicos /robot/<topic>_target y
#           /robot/<topic>_result del nodo correspondiente.
#   seed  : True si el método es iterativo y recibe la pose
#           actual como semilla (mensaje IKJacobTarget).
# ------------------------------------------------------------

IK_METHODS = {
    'alg': {
        'tab': "CI Algebraico",
        'title': "Cinemática inversa - Método algebraico",
        'topic': 'ik',
        'seed': False,
    },
    'newton': {
        'tab': "CI Newton",
        'title': "Cinemática inversa - Método de Newton-Raphson",
        'topic': 'ik_newton',
        'seed': True,
    },
    'des': {
        'tab': "CI Desacople",
        'title': "Cinemática inversa - Método por desacople",
        'topic': 'ik_des',
        'seed': False,
    },
    'geom': {
        'tab': "CI Geométrico",
        'title': "Cinemática inversa - Método geométrico",
        'topic': 'ik_geom',
        'seed': False,
    },
    'grad': {
        'tab': "CI Gradiente descendente",
        'title': "Cinemática inversa - Método de gradiente descendente",
        'topic': 'ik_grad',
        'seed': True,
    },
}


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
        # OFFSETS DE LOS SERVOS
        # ----------------------------------------------------

        self.servo_offset_deg = [
            180.0,    # Hip Roll
            135.0,    # Hip Pitch
            135.0     # Knee
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


        # Publicador utilizado para enviar los ángulos
        # a los servos en grados.

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


        # ====================================================
        # CINEMÁTICA INVERSA (vía ik_*_node, por tópicos)
        #
        # Por cada método: publicador del objetivo, suscriptor
        # del resultado y el resultado pendiente de aplicar en
        # la GUI. El resultado se guarda en on_ik_result (hilo
        # de ROS) y se aplica en refresh() (hilo de Tkinter).
        #
        # last_ik_target guarda el último objetivo (x, y, z)
        # pedido en cada pestaña, para la verificación MTH.
        # ====================================================

        self.ik_target_pubs = {}
        self.ik_result_subs = {}
        self.pending_ik_results = {}
        self.last_ik_target = {}

        for key, method in IK_METHODS.items():

            self.ik_target_pubs[key] = self.create_publisher(
                IKJacobTarget if method['seed'] else Point,
                f"/robot/{method['topic']}_target",
                10
            )

            self.ik_result_subs[key] = self.create_subscription(
                IKResult,
                f"/robot/{method['topic']}_result",
                lambda msg, key=key: self.on_ik_result(key, msg),
                10
            )

            self.pending_ik_results[key] = None
            self.last_ik_target[key] = None


        # Z de "home" según la pierna (espejo en z).

        self.home_z = (
            -HOME_Z_LEFT if self.leg_side == 'right' else HOME_Z_LEFT
        )


        # ====================================================
        # CREACIÓN DE LA INTERFAZ
        # ====================================================

        self.root = tk.Tk()

        self.root.title(
            f"Teleop pata bípedo - Pierna {self.leg_name}"
        )


        # ----------------------------------------------------
        # Tamaño inicial de la ventana
        # ----------------------------------------------------

        self.root.geometry("650x760")

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
            padding=8
        )

        notebook.add(
            tab_fk,
            text="Cinemática directa"
        )

        # El título de la ventana muestra el método de la
        # pestaña activa, para diferenciarlas mejor.

        self.tab_titles = {
            str(tab_fk): "Cinemática directa"
        }

        ik_tabs = {}

        for key, method in IK_METHODS.items():

            ik_tabs[key] = ttk.Frame(
                notebook,
                padding=8
            )

            notebook.add(
                ik_tabs[key],
                text=method['tab']
            )

            self.tab_titles[str(ik_tabs[key])] = method['title']

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
        # MENSAJE DE ESTADO
        # ====================================================

        self.status_label = tk.Label(
            manual_frame,
            text="",
            font=('Arial', 10, 'bold'),
            anchor='center'
        )

        self.status_label.grid(
            row=2,
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
            text="Posición (mm)",
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
            text="x = 0.0000 mm"
        )

        self.position_labels['x'].grid(
            row=0,
            column=0,
            padx=(20, 30),
            pady=2
        )


        self.position_labels['y'] = ttk.Label(
            position_frame,
            text="y = 0.0000 mm"
        )

        self.position_labels['y'].grid(
            row=0,
            column=1,
            padx=30,
            pady=2
        )


        self.position_labels['z'] = ttk.Label(
            position_frame,
            text="z = 0.0000 mm"
        )

        self.position_labels['z'].grid(
            row=0,
            column=2,
            padx=(30, 20),
            pady=2
        )


        # ====================================================
        # PESTAÑAS DE CINEMÁTICA INVERSA
        # ====================================================

        self.ik_widgets = {}

        for key in IK_METHODS:

            self.ik_widgets[key] = self.build_ik_tab(
                ik_tabs[key],
                key
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

    def on_slider_change(
        self,
        index,
        value
    ):

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
        # Mensaje RobotCommand
        # ----------------------------------------------------

        command_msg = RobotCommand()

        command_msg.position = angles_rad

        self.command_pub.publish(
            command_msg
        )


        # ----------------------------------------------------
        # Mensaje para los servos
        # ----------------------------------------------------

        servo_msg = Float32MultiArray()

        servo_msg.data = [
            float(
                self.target_deg[i]
                + self.servo_offset_deg[i]
            )
            for i in range(
                self.num_joints
            )
        ]

        self.servo_pub.publish(
            servo_msg
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
            text=f"x = {x:.4f} mm"
        )

        self.position_labels['y'].config(
            text=f"y = {y:.4f} mm"
        )

        self.position_labels['z'].config(
            text=f"z = {z:.4f} mm"
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
    # CAMBIO DE PESTAÑA: título de la ventana con el método
    # ========================================================

    def on_tab_changed(self, notebook):

        method = self.tab_titles.get(notebook.select(), "")

        self.root.title(
            f"Teleop pata bípedo - Pierna {self.leg_name}"
            + (f" - {method}" if method else "")
        )


    # ========================================================
    # CINEMÁTICA INVERSA - CONSTRUCCIÓN DE UNA PESTAÑA
    #
    # Se llama una vez por método. Devuelve un diccionario con
    # las referencias gráficas de la pestaña (entradas X/Y/Z,
    # etiquetas de q1..q3, verificación MTH y estado).
    # ========================================================

    def build_ik_tab(self, parent, key):

        method = IK_METHODS[key]

        ttk.Label(
            parent,
            text=method['title'],
            style='Method.TLabel'
        ).pack(
            pady=(0, 6)
        )


        # ----------------------------------------------------
        # COORDENADA OBJETIVO
        # ----------------------------------------------------

        coords_frame = ttk.LabelFrame(
            parent,
            text="Coordenada objetivo del pie (mm)",
            padding=8,
            style='Section.TLabelframe'
        )

        coords_frame.pack(
            fill='x',
            pady=(0, 8)
        )

        entries = {}

        home = {'X': HOME_X, 'Y': HOME_Y, 'Z': self.home_z}

        for i, axis in enumerate(('X', 'Y', 'Z')):

            ttk.Label(
                coords_frame,
                text=f"{axis}:"
            ).grid(
                row=0,
                column=i * 2,
                padx=(5, 3),
                pady=5
            )

            entry = tk.Entry(
                coords_frame,
                width=10,
                justify='center'
            )

            entry.insert(
                0,
                f"{home[axis]:.2f}"
            )

            entry.grid(
                row=0,
                column=i * 2 + 1,
                padx=(0, 10),
                pady=5
            )

            entries[axis] = entry

        ttk.Button(
            coords_frame,
            text="Calcular y enviar",
            command=lambda: self.send_ik_target(key)
        ).grid(
            row=1,
            column=0,
            columnspan=3,
            pady=(5, 0)
        )

        tk.Button(
            coords_frame,
            text="Home",
            bg='#ffdddd',
            command=lambda: self.go_home(key)
        ).grid(
            row=1,
            column=3,
            columnspan=3,
            pady=(5, 0)
        )

        if method['seed']:

            ttk.Label(
                coords_frame,
                text="Semilla: la pose articular actual "
                     "(pestaña de cinemática directa)."
            ).grid(
                row=2,
                column=0,
                columnspan=6,
                pady=(5, 0)
            )


        # ----------------------------------------------------
        # RESULTADO (q1, q2, q3)
        # ----------------------------------------------------

        result_frame = ttk.LabelFrame(
            parent,
            text="Resultado",
            padding=8,
            style='Section.TLabelframe'
        )

        result_frame.pack(
            fill='x',
            pady=(0, 8)
        )

        q_labels = []

        for i, name in enumerate(
            ('q1 (Hip Roll)', 'q2 (Hip Pitch)', 'q3 (Knee)')
        ):

            ttk.Label(
                result_frame,
                text=f"{name}:",
                width=16
            ).grid(
                row=i,
                column=0,
                padx=(5, 5),
                pady=3,
                sticky='w'
            )

            value_label = ttk.Label(
                result_frame,
                text="--",
                width=10
            )

            value_label.grid(
                row=i,
                column=1,
                padx=(0, 5),
                pady=3,
                sticky='w'
            )

            q_labels.append(
                value_label
            )


        # ----------------------------------------------------
        # VERIFICACIÓN MTH DEL RESULTADO
        # ----------------------------------------------------

        mth = self.build_mth_verification(parent)


        # ----------------------------------------------------
        # MENSAJE DE ESTADO
        # ----------------------------------------------------

        status_label = tk.Label(
            parent,
            text="Ingrese una coordenada y presione \"Calcular y enviar\".",
            font=('Arial', 10, 'bold'),
            wraplength=560,
            justify='center'
        )

        status_label.pack(
            pady=(5, 0)
        )

        return {
            'entries': entries,
            'q_labels': q_labels,
            'mth': mth,
            'status_label': status_label,
        }


    # ========================================================
    # CINEMÁTICA INVERSA - ENVIAR OBJETIVO
    # ========================================================

    def send_ik_target(self, key):

        entries = self.ik_widgets[key]['entries']

        try:
            x = float(entries['X'].get())
            y = float(entries['Y'].get())
            z = float(entries['Z'].get())

        except ValueError:

            self.set_ik_status(
                key,
                "Error: X, Y, Z deben ser valores numéricos.",
                "red"
            )

            return

        self.last_ik_target[key] = (x, y, z)

        if IK_METHODS[key]['seed']:

            # La semilla no se pide al usuario: se toma la pose
            # articular actual (self.target_deg), la misma que se
            # ve en la pestaña de cinemática directa.

            target = IKJacobTarget()
            target.x = x
            target.y = y
            target.z = z
            target.q1_seed = float(self.target_deg[0])
            target.q2_seed = float(self.target_deg[1])
            target.q3_seed = float(self.target_deg[2])

        else:

            target = Point()
            target.x = x
            target.y = y
            target.z = z

        self.ik_target_pubs[key].publish(target)

        self.set_ik_status(
            key,
            "Calculando...",
            "gray"
        )


    # ========================================================
    # CINEMÁTICA INVERSA - IR A HOME (q = 0)
    # ========================================================

    def go_home(self, key):

        entries = self.ik_widgets[key]['entries']

        for axis, value in (
            ('X', HOME_X),
            ('Y', HOME_Y),
            ('Z', self.home_z)
        ):

            entries[axis].delete(0, tk.END)
            entries[axis].insert(0, f"{value:.2f}")

        self.send_ik_target(key)


    # ========================================================
    # CINEMÁTICA INVERSA - RECEPCIÓN DEL RESULTADO
    # (hilo de ROS; solo guarda el dato, ver refresh())
    # ========================================================

    def on_ik_result(self, key, msg):

        self.pending_ik_results[key] = msg


    # ========================================================
    # CINEMÁTICA INVERSA - APLICAR RESULTADO
    # (hilo de Tkinter, llamado desde refresh())
    # ========================================================

    def apply_ik_result(self, key, result):

        widgets = self.ik_widgets[key]

        if not result.reachable or len(result.position) < 3:

            for label in widgets['q_labels']:
                label.config(text="--", foreground='')

            self.set_ik_status(
                key,
                "✗ No convergió / posición no alcanzable."
                if IK_METHODS[key]['seed']
                else "✗ Posición no alcanzable.",
                "red"
            )

            self.reset_mth_verification(widgets['mth'])

            return

        q_deg = [math.degrees(q) for q in result.position]

        self.show_ik_q_labels(widgets['q_labels'], q_deg)

        if self.last_ik_target[key] is not None:

            x_obj, y_obj, z_obj = self.last_ik_target[key]

            self.update_mth_verification(
                widgets['mth'],
                result.position[0], result.position[1], result.position[2],
                x_obj, y_obj, z_obj
            )


        # ----------------------------------------------------
        # Una solución fuera de los límites articulares no se
        # aplica: los sliders la recortarían y el pie no
        # llegaría al objetivo (y se mandaría a los servos).
        # ----------------------------------------------------

        fuera = self.ik_limit_violations(q_deg)

        if fuera:

            self.set_ik_status(
                key,
                "⚠ Solución fuera de los límites articulares: "
                + ", ".join(fuera)
                + ". No se aplicó.",
                "dark orange"
            )

            return


        # ----------------------------------------------------
        # Movemos el robot por el mismo camino que la pestaña
        # de cinemática directa, y sincronizamos sliders,
        # entradas y matriz con la nueva pose.
        # ----------------------------------------------------

        for i in range(min(self.num_joints, len(q_deg))):

            self.target_deg[i] = q_deg[i]

            self.sliders[i].set(q_deg[i])

            self.slider_labels[i].config(
                text=f"{q_deg[i]:.1f}°"
            )

            self.entries[i].delete(0, tk.END)
            self.entries[i].insert(0, f"{q_deg[i]:.1f}")

        self.publish_target()

        self.update_kinematics()

        self.set_ik_status(
            key,
            "✓ Movimiento realizado.",
            "green"
        )


    # ========================================================
    # CINEMÁTICA INVERSA - LÍMITES ARTICULARES DEL RESULTADO
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


    # ========================================================
    # CINEMÁTICA INVERSA - MENSAJE DE ESTADO
    # ========================================================

    def set_ik_status(self, key, message, color):

        self.ik_widgets[key]['status_label'].config(
            text=message,
            foreground=color
        )


    # ========================================================
    # VERIFICACIÓN MTH - CONSTRUCCIÓN DEL WIDGET
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
    # Arma Tdes a partir del (q1, q2, q3) que calculó el método
    # de la pestaña, le pide al método MTH que recupere esos
    # mismos ángulos leyendo la matriz, y muestra la
    # comparación. El método MTH está escrito para la pierna
    # izquierda; para la derecha se compara contra el objetivo
    # espejado en z y se muestra la T04 de la pierna derecha.
    # ========================================================

    def update_mth_verification(
        self, mth_widgets, q1, q2, q3, x_obj, y_obj, z_obj
    ):

        if self.leg_side == 'right':
            z_obj = -z_obj

        resultado = verificar_con_mth(q1, q2, q3, x_obj, y_obj, z_obj)

        if self.leg_side == 'right':
            T = forward_kinematics_right(
                [math.degrees(q) for q in (q1, q2, q3)]
            )[3]
        else:
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

        except Exception as e:

            self.get_logger().error(
                f"Error procesando JointState: {e}"
            )


    # ========================================================
    # ACTUALIZACIÓN PERIÓDICA DE LA GUI
    # ========================================================

    def refresh(self):

        # ----------------------------------------------------
        # Si llegó un resultado nuevo de cinemática inversa,
        # lo aplicamos acá (hilo de Tkinter), nunca dentro de
        # on_ik_result (hilo de ROS).
        #
        # Un error al aplicar un resultado NO debe cortar este
        # ciclo: si no se reprograma, la interfaz deja de
        # aplicar resultados para siempre.
        # ----------------------------------------------------

        for key in IK_METHODS:

            pendiente = self.pending_ik_results[key]

            if pendiente is None:
                continue

            self.pending_ik_results[key] = None

            try:

                self.apply_ik_result(key, pendiente)

            except Exception as e:

                self.get_logger().error(
                    f"Error aplicando resultado de IK ({key}): {e}"
                )

        self.root.after(
            50,
            self.refresh
        )


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
