from setuptools import find_packages, setup

package_name = 'obs_avoid'

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
            'ref_gov_target = obs_avoid.ref_gov_target:main',
            'pid_control = obs_avoid.pid_control:main',
        ],
    },
)
