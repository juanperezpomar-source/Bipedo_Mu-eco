# ============================================================
# NODO DE ENCODERS DE LOS SERVOS (muñeco)
# ============================================================
#
# Traduce y vigila lo que reporta la ESP32 (firmware
# muneco_motores) sobre los encoders AS5600 de cada articulación.
#
# Entradas (desde la ESP32 por micro-ROS):
#   /servo_states   Float32MultiArray  ángulo medido + offset [°]
#   /encoder_raw    Float32MultiArray  lectura cruda 0..4095 (-1 = no responde)
#   /servo_commands Float32MultiArray  último objetivo (ángulo + offset) [°]
#
# Salidas:
#   /robot/encoder_joint_states  sensor_msgs/JointState
#       Posición REAL de cada articulación [rad], sin offsets.
#       NaN en la articulación cuyo encoder no responde (la ESP32
#       en ese caso manda la posición de su rampa, no una medida).
#   /robot/encoder_status        std_msgs/String
#       Estado por articulación separado por comas, en orden
#       roll, pitch, rodilla:
#           ok             encoder leyendo
#           sin_respuesta  el AS5600 no responde (raw = -1)
#           sin_esp        la ESP32 no publica hace más de timeout_sec
#   /robot/encoder_error         Float32MultiArray
#       Objetivo comandado - medido [°] (NaN si no hay dato).
#
# ============================================================

import math
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, String


class ServoEncodersNode(Node):

    def __init__(self):

        super().__init__('servo_encoders')

        self.declare_parameter(
            'joint_names',
            [
                'Left_Hip_Roll_Joint',
                'Left_Hip_Pitch_Joint',
                'Left_Knee_Joint'
            ]
        )

        # Deben coincidir con los offsets del firmware y de teleop_node.
        self.declare_parameter('servo_offset_deg', [180.0, 135.0, 135.0])

        self.declare_parameter('timeout_sec', 0.5)

        self.joint_names = list(self.get_parameter('joint_names').value)
        self.offsets = [float(v) for v in self.get_parameter('servo_offset_deg').value]
        self.timeout_sec = float(self.get_parameter('timeout_sec').value)
        self.n = len(self.joint_names)

        self.medido_deg = [math.nan] * self.n
        self.objetivo_deg = [math.nan] * self.n
        self.raw_ok = [False] * self.n
        self.t_estado = None
        self.t_raw = None
        self.ultimo_status = None

        self.create_subscription(
            Float32MultiArray, '/servo_states', self.on_servo_states, 10)
        self.create_subscription(
            Float32MultiArray, '/encoder_raw', self.on_encoder_raw, 10)
        self.create_subscription(
            Float32MultiArray, '/servo_commands', self.on_servo_commands, 10)

        self.joint_pub = self.create_publisher(
            JointState, '/robot/encoder_joint_states', 10)
        self.status_pub = self.create_publisher(
            String, '/robot/encoder_status', 10)
        self.error_pub = self.create_publisher(
            Float32MultiArray, '/robot/encoder_error', 10)

        self.create_timer(0.2, self.publicar_status)

        self.get_logger().info(
            f'servo_encoders listo: {self.joint_names}, offsets {self.offsets}')

    # ------------------------------------------------------------

    def on_encoder_raw(self, msg):
        for i in range(min(self.n, len(msg.data))):
            self.raw_ok[i] = msg.data[i] >= 0
        self.t_raw = time.monotonic()

    def on_servo_commands(self, msg):
        for i in range(min(self.n, len(msg.data))):
            self.objetivo_deg[i] = float(msg.data[i]) - self.offsets[i]

    def on_servo_states(self, msg):
        if len(msg.data) < self.n:
            return
        self.t_estado = time.monotonic()
        raw_reciente = self._reciente(self.t_raw)

        for i in range(self.n):
            if raw_reciente and self.raw_ok[i]:
                self.medido_deg[i] = float(msg.data[i]) - self.offsets[i]
            else:
                self.medido_deg[i] = math.nan

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.joint_names
        js.position = [math.radians(v) for v in self.medido_deg]
        self.joint_pub.publish(js)

        self.error_pub.publish(Float32MultiArray(data=[
            o - m for o, m in zip(self.objetivo_deg, self.medido_deg)
        ]))

    # ------------------------------------------------------------

    def _reciente(self, t):
        return t is not None and time.monotonic() - t <= self.timeout_sec

    def publicar_status(self):
        if not (self._reciente(self.t_estado) and self._reciente(self.t_raw)):
            estados = ['sin_esp'] * self.n
        else:
            estados = ['ok' if ok else 'sin_respuesta' for ok in self.raw_ok]

        status = ','.join(estados)
        self.status_pub.publish(String(data=status))

        if status != self.ultimo_status:
            log = self.get_logger().info if status == ','.join(['ok'] * self.n) \
                else self.get_logger().warn
            log(f'Encoders: {status}')
            self.ultimo_status = status


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ServoEncodersNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
