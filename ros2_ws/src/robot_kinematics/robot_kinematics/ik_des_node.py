# ============================================================
# NODO DE CINEMÁTICA INVERSA (MÉTODO DE DESACOPLE)
# ============================================================
#
# Recibe un punto objetivo (x, y, z) [mm] en /robot/ik_des_target y
# publica en /robot/ik_des_result:
#
#   - reachable: si el punto es alcanzable geométricamente.
#   - position:  q1, q2, q3 [rad] para alcanzarlo (si aplica).
#
# Los métodos están escritos para la pierna IZQUIERDA. La
# pierna derecha es su espejo en z (mismos q1, q2, q3 llevan el
# pie a (x, y, -z)), así que con leg_side='right' se resuelve el
# objetivo (x, y, -z) con el mismo método.
#
# ============================================================

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from robot_interfaces.msg import IKResult

from robot_kinematics.kinem_invers_leg_mth_desacople_MUNECO_izq import (
    cinematica_inversa_pata_des_xyz,
    joint_limit_warnings,
)


class IKDesNode(Node):

    def __init__(self):
        super().__init__('ik_des_node')

        self.declare_parameter('leg_side', 'left')
        self.leg_side = (
            self.get_parameter('leg_side')
            .get_parameter_value()
            .string_value
            .lower()
        )

        self.target_sub = self.create_subscription(
            Point, '/robot/ik_des_target', self.on_target, 10)

        self.result_pub = self.create_publisher(
            IKResult, '/robot/ik_des_result', 10)

        self.get_logger().info(
            f'ik_des_node listo (pierna {self.leg_side})')

    def on_target(self, msg):
        # Objetivo expresado en el sistema de la pierna izquierda.
        z = -msg.z if self.leg_side == 'right' else msg.z

        q1, q2, q3, alcanzable = cinematica_inversa_pata_des_xyz(
            msg.x, msg.y, z)

        result = IKResult()
        result.reachable = bool(alcanzable)

        if alcanzable:
            result.position = [float(q1), float(q2), float(q3)]

            q_deg = np.degrees([q1, q2, q3])
            for warning in joint_limit_warnings(*q_deg):
                self.get_logger().warn(warning)
        else:
            result.position = []
            self.get_logger().warn(
                f'Objetivo ({msg.x:.1f}, {msg.y:.1f}, {msg.z:.1f}) '
                f'no alcanzable.'
            )

        self.result_pub.publish(result)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(IKDesNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
