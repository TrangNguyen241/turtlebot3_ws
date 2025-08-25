from setuptools import find_packages, setup

package_name = 'lya_lqr_control'

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
            'lya_lqr_target = lya_lqr_control.lya_lqr_target:main', 
            'lya_lqr_traj = lya_lqr_control.lya_lqr_traj:main'

        ],
    },
)
