"""Pruebas de perfiles_temporales con los ejemplos resueltos de la clase
"Manipulator Inverse kinematics II" (diapositivas 47-73)."""
import math

import numpy as np

from robot_kinematics import perfiles_temporales as pt


def coef_global(perfil, k=0):
    return perfil.coeficientes_globales()[k][2]


def test_cubico_5_a_25_en_2s():                       # diap. 47
    p = pt.cubico(5, 25, 0, 2)
    assert np.allclose(coef_global(p), [5, 0, 15, -5])


def test_cubico_de_t2_a_t4():                         # diap. 49
    p = pt.cubico(10, 60, 2, 4)
    assert np.allclose(coef_global(p), [260, -300, 112.5, -12.5])
    _, v, a = p.evaluar(np.array([2.0, 3.0, 4.0]))
    assert np.allclose(v, [0, 37.5, 0])
    assert math.isclose(a[0], 75) and math.isclose(a[-1], -75)


def test_quintico_de_t2_a_t4():                       # diap. 51
    p = pt.quintico(10, 60, 2, 4)
    assert np.allclose(coef_global(p), [-1540, 3000, -2250, 812.5, -140.625, 9.375])
    _, v, a = p.evaluar(np.array([2.0, 4.0]))
    assert np.allclose(v, 0) and np.allclose(a, 0)     # sin saltos de aceleración


def test_cubico_con_punto_intermedio():               # diap. 65
    p = pt.cubico_puntos([5, 15, 20], [0, 1, 2], vs=[0, 5, 0])
    assert np.allclose(p.tramos[0][2], [5, 0, 25, -15])
    assert np.allclose(p.tramos[1][2], [15, 5, 5, -5])


def test_velocidades_de_paso():                       # diap. 64
    vs = pt.velocidades_de_paso([0, 2, 6, 3, 1], [0, 1, 2, 3, 4])
    assert vs == [0.0, 3.0, 0.0, -2.5, 0.0]


def test_ley_temporal_camino():
    pts = np.array([[0, 0, 0], [30, 0, 0], [30, 40, 0]], float)
    t2, P2 = pt.ley_temporal_camino(pts, 'quintico', 20.0)
    v2 = np.linalg.norm(np.diff(P2, axis=0), axis=1) / np.diff(t2)
    assert v2.max() <= 20.0 + 0.1
    assert np.allclose(P2[0], pts[0]) and np.allclose(P2[-1], pts[-1])
