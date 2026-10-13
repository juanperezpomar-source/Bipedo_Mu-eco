"""kinem_muneco trabaja en el sistema de las pestañas de IK del muñeco:
su cinemática directa debe dar el mismo pie que
cinematica_directa_der_MUNECO_izq, y su cinemática inversa debe volver al
mismo punto."""

import numpy as np
import pytest

from robot_kinematics import kinem_muneco as K
from robot_kinematics import cinematica_directa_der_MUNECO_izq as G

FK_EQUIPO = {'left': G.forward_kinematics_left, 'right': G.forward_kinematics_right}


@pytest.mark.parametrize('side', ['left', 'right'])
def test_mismo_pie_que_cinematica_del_equipo(side):
    rng = np.random.default_rng(0)
    for t in rng.uniform([0, -90, -90], [90, 90, 90], (300, 3)):
        equipo = G.get_position(FK_EQUIPO[side](list(t))[-1])
        assert np.allclose(K.posicion_pie(t, side), equipo, atol=1e-6)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_ida_y_vuelta_ik(side):
    rng = np.random.default_rng(1)
    for t in rng.uniform([0, -90, -90], [90, 90, 90], (300, 3)):
        p = K.posicion_pie(t, side)
        t_ik, ok, _ = K.ik_geometrico(*p, side=side, semilla_deg=t)
        assert ok
        assert np.allclose(K.posicion_pie(t_ik, side), p, atol=1e-6)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_papel_recomendado_horizontal_y_alcanzable(side):
    papel = K.PAPEL_RECOMENDADO[side]
    x = papel['x_papel']
    y0, z0 = papel['centro']
    # El papel queda por encima del pie colgando (X apunta hacia abajo)
    assert K.posicion_home(side)[0] > x
    h = papel['lado'] / 2
    for dy in (-h, 0, h):
        for dz in (-h, 0, h):
            for alto in (0.0, papel['levantar']):
                t, ok, _ = K.ik_geometrico(x - alto, y0 + dy, z0 + dz, side=side)
                assert ok and t[2] > 0          # alcanzable, rodilla en flexión
