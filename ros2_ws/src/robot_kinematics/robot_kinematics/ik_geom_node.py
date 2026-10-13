# ============================================================
# NODO DE CINEMÁTICA INVERSA (MÉTODO GEOMÉTRICO)
# ============================================================
#
# Recibe un punto objetivo (x, y, z) [mm] en /robot/ik_geom_target y
# publica en /robot/ik_geom_result:
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
from std_msgs.msg import Float64MultiArray
from robot_interfaces.msg import IKResult

from robot_kinematics.kinem_invers_leg_Geometrico_MUNECO_izq import (
    cinematica_inversa_pata_geom,
    joint_limit_warnings,
    L1, L2, L3, L4, L5,
)


class IKGeomNode(Node):

    def __init__(self):
        super().__init__('ik_geom_node')

        self.declare_parameter('leg_side', 'left')
        self.leg_side = (
            self.get_parameter('leg_side')
            .get_parameter_value()
            .string_value
            .lower()
        )

        self.target_sub = self.create_subscription(
            Point, '/robot/ik_geom_target', self.on_target, 10)

        # Canal adicional: [x_mm, y_mm, z_mm, rama]; rama +1 arriba, -1 abajo.
        self.branch_target_sub = self.create_subscription(
            Float64MultiArray, '/robot/ik_geom_target_branch', self.on_branch_target, 10)

        self.result_pub = self.create_publisher(
            IKResult, '/robot/ik_geom_result', 10)

        self.get_logger().info(
            f'ik_geom_node listo (pierna {self.leg_side})')

    def on_branch_target(self, msg):
        if len(msg.data) != 4 or not all(np.isfinite(v) for v in msg.data):
            self.get_logger().error(
                'Solicitud IK inválida: se requieren [x, y, z, rama] finitos.'
            )
            return
        x, y, z, signo = msg.data
        if signo not in (1.0, -1.0):
            self.get_logger().error('La rama debe ser +1 (arriba) o -1 (abajo).')
            return
        self.resolver_objetivo(x, y, z, 'arriba' if signo == 1.0 else 'abajo')

    def on_target(self, msg):
        # Compatibilidad con publicadores anteriores: Point implica codo arriba.
        self.resolver_objetivo(msg.x, msg.y, msg.z, 'arriba')

    def resolver_objetivo(self, x, y, z, codo):
        # Objetivo expresado en el sistema de la pierna izquierda.
        z_izq = -z if self.leg_side == 'right' else z

        q1, q2, q3, alcanzable = cinematica_inversa_pata_geom(
            x, y, z_izq, L1, L2, L3, L4, L5, codo=codo)

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
                f'Objetivo ({x:.1f}, {y:.1f}, {z:.1f}) '
                f'no alcanzable.'
            )

        self.result_pub.publish(result)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(IKGeomNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
