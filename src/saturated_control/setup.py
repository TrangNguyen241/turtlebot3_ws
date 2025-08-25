from setuptools import find_packages, setup

package_name = 'saturated_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nguyehtt',
    maintainer_email='nguyehtt@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'saturated_single = saturated_control.saturated_single:main',
            'saturated_multiple = saturated_control.saturated_multiple:main',
            'saturated_traj = saturated_control.saturated_traj:main',
            'sat_lya_target = saturated_control.sat_lya_target:main',
            'lya_pid_target = saturated_control.lya_pid_target:main',
            'lya_lqr_target = saturated_control.lya_lqr_target:main'
        ],
    },
)
