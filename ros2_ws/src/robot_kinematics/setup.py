from glob import glob

from setuptools import find_packages, setup

package_name = 'robot_kinematics'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/trayectorias',
            glob('trayectorias/*.txt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='david',
    maintainer_email='david.collazos_her@uao.edu.co',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'kinematics_node = robot_kinematics.kinematics_node:main',
            'ik_node = robot_kinematics.ik_node:main',
            'ik_newton_node = robot_kinematics.ik_newton_node:main',
            'ik_des_node = robot_kinematics.ik_des_node:main',
            'ik_geom_node = robot_kinematics.ik_geom_node:main',
            'ik_grad_node = robot_kinematics.ik_grad_node:main',
        ],
    },
)
