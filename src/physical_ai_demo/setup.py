import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'physical_ai_demo'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'assets'), glob('assets/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='OpenCV and RealSense perception driving a virtual robot through Corebrum Synapse.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_eye = physical_ai_demo.aruco_eye:main',
            'marker_skill = physical_ai_demo.marker_skill:main',
            'stage_base = physical_ai_demo.stage_base:main',
        ],
    },
)
