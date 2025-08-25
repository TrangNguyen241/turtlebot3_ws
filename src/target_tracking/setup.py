from setuptools import find_packages, setup

package_name = 'target_tracking'

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
            'target_tracking_controllers = target_tracking.target_tracking_controllers:main',
            'velocity_threshold_checker = target_tracking.velocity_threshold_checker:main', 
            'test_max_to_min_velocity = target_tracking.test_max_to_min_velocity:main'
        ],
    },
)
