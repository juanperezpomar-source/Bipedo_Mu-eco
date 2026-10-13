# ============================================================
# NODO DE CONTROL (CAPA BAJA)
# ============================================================
#
# MODO VERIFICACION:
#   Recibe la consigna desde /robot/joint_targets,
#   aplica limites articulares y publica el estado tal cual
#   en /joint_states.
#
#   Sin PID, sin motor model, sin filtros. Sirve para confirmar
#   que lo que entra por los sliders se refleja en RViz.
#
# MODO PID:
#   Cuando verify_mode=false, se puede volver al modo PID para
#   control avanzado (trayectorias, IK, etc.).
#
# Uso:
#   ros2 param set /control_node verify_mode true
#
# ============================================================


import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool

from robot_interfaces.msg import JointTarget, JointState

from sensor_msgs.msg import JointState as SensorJointState


class ControlNode(Node):

    def __init__(self):

        super().__init__('control_node')


        # ====================================================
        # CONFIGURACIÓN GENERAL
        # ====================================================

        self.declare_parameter(
            'num_joints',
            3
        )


        # Default = pierna derecha
        # (Right_Hip_Roll/Pitch + Right_Knee, ver
        # robot_description/urdf/urdf_der).
        #
        # bringup_izq.launch.py sobreescribe esto con los
        # joints Left_* al lanzar la pata izquierda.

        self.declare_parameter(
            'joint_names',
            [
                'Right_Hip_Roll_Joint',
                'Right_Hip_Pitch_Joint',
                'Right_Knee_Joint'
            ]
        )


        # ====================================================
        # LIMITES ARTICULARES
        # ====================================================
        #
        # Limite INTERNO real del motor, en RADIANES.
        #
        # Ya no lo sobreescribe bringup_*.launch.py:
        # este .py es la unica fuente de verdad.
        #
        # Es un clamp de seguridad puertas adentro:
        # el operador no lo ve ni lo toca desde la interfaz
        # (eso lo hace robot_teleop, ver
        # joint_limits_lower_deg/upper_deg en teleop_node.py,
        # que debe reflejar estos mismos valores en grados).
        #
        # Para cambiarlo:
        #
        #   editar estas listas
        #
        # o:
        #
        #   ros2 param set /control_node joint_limits_lower [...]
        #   ros2 param set /control_node joint_limits_upper [...]
        #
        # Se relee cada ciclo.
        #
        # cadera-roll (indice 0) hoy NO es el limite nominal
        # del URDF (-0.34907 rad / -20 grados):
        # esta en -1.91986 rad (-110 grados) por un offset
        # fisico temporal del motor.
        #
        # Cuando se recalibre, volver a poner -0.34907 aca
        # (y el equivalente en teleop_node.py).
        #
        # ====================================================

        self.declare_parameter(
            'joint_limits_lower',
            [
                -0.261799,
                -1.5708,
                -1.5708
            ]
        )

        self.declare_parameter(
            'joint_limits_upper',
            [
                1.5708,
                1.5708,
                1.5708
            ]
        )


        # ====================================================
        # MODO DE VERIFICACIÓN
        # ====================================================
        #
        # True = clamp y publicar directo.
        #
        # False = utilizar PID para control avanzado.
        #
        # ====================================================

        self.declare_parameter(
            'verify_mode',
            True
        )


        # ====================================================
        # INVERSIÓN DEL KNEE PITCH HACIA RVIZ (opcional)
        # ====================================================
        #
        # False (por defecto): RViz recibe el mismo ángulo que la
        # GUI y la cinemática. Comparando el URDF del muñeco
        # (Muneco_Bipedo_URDF_V3) con la tabla DH
        # (cinematica_directa_der_MUNECO_izq), con +Knee Pitch los
        # dos llevan el pie hacia ATRÁS (la rodilla se recoge) y
        # el pie coincide a ~1 mm en la zona de dibujo. Invertir
        # el signo hacía que RViz doblara la rodilla al revés que
        # la cinemática inversa (el pie se iba hacia adelante) y
        # el rastro de IK/trayectorias no caía sobre el pie.
        #
        # True: vuelve al comportamiento anterior (signo invertido
        # solo hacia /joint_states y /robot/joint_commands).
        #
        # /robot/joint_states (estado interno que lee la GUI)
        # nunca se invierte.
        # ====================================================

        self.declare_parameter(
            'invert_knee_rviz',
            False
        )


        # ====================================================
        # LECTURA INICIAL DE PARÁMETROS
        # ====================================================

        self.n = self.get_parameter(
            'num_joints'
        ).value

        self.joint_names = self.get_parameter(
            'joint_names'
        ).value

        self.lower = self.get_parameter(
            'joint_limits_lower'
        ).value

        self.upper = self.get_parameter(
            'joint_limits_upper'
        ).value

        self.verify_mode = self.get_parameter(
            'verify_mode'
        ).value


        # ====================================================
        # SUSCRIPTORES
        # ====================================================

        self.sub = self.create_subscription(
            JointTarget,
            '/robot/joint_targets',
            self.on_target,
            10
        )


        self.estop_sub = self.create_subscription(
            Bool,
            '/robot/e_stop',
            self.on_estop,
            10
        )


        # ====================================================
        # PUBLICADORES
        # ====================================================

        # Estado interno del robot.

        self.pub = self.create_publisher(
            JointState,
            '/robot/joint_states',
            10
        )


        # Estado estándar de ROS.
        #
        # Este tópico es utilizado por robot_state_publisher
        # para generar los TF que posteriormente utiliza RViz.

        self.std_pub = self.create_publisher(
            SensorJointState,
            '/joint_states',
            10
        )


        # Comando hacia simulación / actuadores.

        self.cmd_pub = self.create_publisher(
            JointTarget,
            '/robot/joint_commands',
            10
        )


        # ====================================================
        # TEMPORIZADOR
        # ====================================================
        #
        # 0.02 s = 20 ms = 50 Hz
        #
        # Actualmente es suficiente para el modo de verificación.
        #
        # ====================================================

        self.timer = self.create_timer(
            0.02,
            self.control_loop
        )


        # ====================================================
        # VARIABLES DE ESTADO
        # ====================================================

        self.target = None

        self.e_stop = False

        self._last_clamped = [
            0.0
        ] * self.n


        # ====================================================
        # INFORMACIÓN DE INICIO
        # ====================================================

        self.get_logger().info(
            f'control_node listo | '
            f'modo={"VERIFICACION" if self.verify_mode else "PID"} | '
            f'joints={self.n}'
        )

        if self.get_parameter('invert_knee_rviz').value:
            self.get_logger().info(
                'Knee Pitch: signo invertido antes de publicar '
                'a RViz y simulación (invert_knee_rviz=True)'
            )


    # ========================================================
    # RECEPCIÓN DE CONSIGNA
    # ========================================================

    def on_target(self, msg):
        """
        Guarda la consigna recibida desde el nodo de cinemática.

        La consigna llega utilizando la convención de ángulos
        definida para el robot.

        En particular:

            Knee Pitch positivo = flexionar/recoger rodilla.

        La inversión necesaria para el sentido físico del
        actuador y del URDF se realiza posteriormente en
        control_loop().
        """

        self.target = msg


    # ========================================================
    # RECEPCIÓN DE E-STOP
    # ========================================================

    def on_estop(self, msg):
        """
        Actualiza el estado del paro de emergencia.
        """

        self.e_stop = msg.data


    # ========================================================
    # BUCLE PRINCIPAL DE CONTROL
    # ========================================================

    def control_loop(self):

        # ----------------------------------------------------
        # Leer limites y modo cada ciclo para que sean
        # dinamicos.
        # ----------------------------------------------------

        self.lower = self.get_parameter(
            'joint_limits_lower'
        ).value

        self.upper = self.get_parameter(
            'joint_limits_upper'
        ).value

        self.verify_mode = self.get_parameter(
            'verify_mode'
        ).value


        # ====================================================
        # CREAR ESTADO
        # ====================================================

        state = JointState()

        state.name = self.joint_names


        # ====================================================
        # E-STOP
        # ====================================================

        if self.e_stop:

            state.position = [
                0.0
            ] * self.n

            state.velocity = [
                0.0
            ] * self.n

            state.effort = [
                0.0
            ] * self.n

            state.battery = 100.0

            self.pub.publish(
                state
            )

            self.publish_std(
                state.position,
                state.effort
            )

            return


        # ====================================================
        # EXISTE UNA CONSIGNA
        # ====================================================

        if self.target is not None:

            goal = (
                list(self.target.position)
                +
                [0.0] * self.n
            )

            goal = goal[
                :self.n
            ]


            # =================================================
            # MODO VERIFICACIÓN
            # =================================================

            if self.verify_mode:

                # ------------------------------------------------
                # MODO VERIFICACION:
                # clamp y publicar directo.
                # ------------------------------------------------

                clamped = [
                    max(
                        self.lower[i],
                        min(
                            self.upper[i],
                            goal[i]
                        )
                    )

                    for i in range(self.n)
                ]


                state.position = clamped

                state.effort = [
                    0.0
                ] * self.n


            # =================================================
            # MODO PID
            # =================================================

            else:

                # ------------------------------------------------
                # MODO PID
                # (para cuando quieras agregar cinematica inversa
                # o trayectorias mas adelante).
                # ------------------------------------------------

                from robot_control.pid_controllers import (
                    PIDController
                )

                from robot_control.motor_control import (
                    MotorController
                )


                if not hasattr(
                    self,
                    'pid'
                ):

                    self.pid = PIDController(
                        self.n
                    )

                    self.motor = MotorController(
                        self.n
                    )

                    self.estimated = [
                        0.0
                    ] * self.n


                correction = self.pid.update(
                    self.estimated,
                    goal,
                    0.02
                )


                command = self.motor.apply(
                    goal,
                    correction
                )


                for i in range(
                    self.n
                ):

                    self.estimated[i] = max(
                        self.lower[i],
                        min(
                            self.upper[i],
                            self.estimated[i]
                            +
                            command[i] * 0.02
                        )
                    )


                state.position = self.estimated

                state.effort = command


        else:

            # =================================================
            # SIN TARGET
            # =================================================
            #
            # Mantener ultima posicion clamped.
            # =================================================

            state.position = list(
                getattr(
                    self,
                    '_last_clamped',
                    [0.0] * self.n
                )
            )

            state.effort = [
                0.0
            ] * self.n


        # ====================================================
        # INFORMACIÓN ADICIONAL DEL ESTADO
        # ====================================================

        state.velocity = [
            0.0
        ] * self.n

        state.battery = 100.0


        # ====================================================
        # GUARDAR ÚLTIMA POSICIÓN
        # ====================================================
        #
        # Guardamos la posición antes de invertir el Knee Pitch.
        #
        # Esto permite que internamente el control siga trabajando
        # con la convención original del robot.
        #
        # ====================================================

        self._last_clamped = list(
            state.position
        )


        # ====================================================
        # POSICIÓN HACIA RVIZ / SIMULACIÓN
        # ====================================================
        #
        # Igual que la de la GUI, salvo que invert_knee_rviz=True
        # (ver la declaración del parámetro): en ese caso se
        # invierte solo el índice 2 (Knee Pitch).
        #
        # ====================================================

        visual = list(state.position)

        if (
            self.n >= 3
            and self.get_parameter('invert_knee_rviz').value
        ):

            visual[2] = -visual[2]


        # ====================================================
        # PUBLICAR ESTADO
        # ====================================================
        #
        # /robot/joint_states lleva la convención de la GUI;
        # /joint_states lleva "visual" (invert_knee_rviz).
        #
        # /joint_states -> robot_state_publisher -> RViz
        #
        #
        # ====================================================

        self.pub.publish(
            state
        )


        self.publish_std(
            visual,
            state.effort
        )


        # ====================================================
        # PUBLICAR COMANDO HACIA SIMULACIÓN / ACTUADORES
        # ====================================================
        #
        # sim_bridge simplemente copia este valor hacia
        # /joint_group_position_controller/commands.
        #
        # Por eso NO es necesario modificar sim_bridge.
        #
        # Sale con la misma posición que RViz ("visual").
        #
        # ====================================================

        cmd = JointTarget()

        cmd.position = list(
            visual
        )

        cmd.velocity = [
            0.0
        ] * self.n

        self.cmd_pub.publish(
            cmd
        )


    # ========================================================
    # PUBLICAR SENSOR_JOINT_STATE
    # ========================================================

    def publish_std(
        self,
        position,
        effort
    ):
        """
        Publica el estado utilizando el mensaje estándar
        de ROS.

        Este tópico:

            /joint_states

        es utilizado por robot_state_publisher para generar
        los TF que RViz utiliza para visualizar el robot.

        La posición recibida aquí es la de RViz ("visual",
        ver invert_knee_rviz).
        """

        msg = SensorJointState()

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        msg.header.frame_id = 'Base_link'

        msg.name = self.joint_names

        msg.position = list(
            position
        )

        msg.velocity = [
            0.0
        ] * self.n

        msg.effort = list(
            effort
        )

        self.std_pub.publish(
            msg
        )


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = ControlNode()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == '__main__':

    main()
