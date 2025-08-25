from setuptools import find_packages, setup

package_name = 'cbf_avoid_colli'

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
            'cbf_avoid_colli = cbf_avoid_colli.cbf_avoid_colli:main',
            'cbf_mov_static_obs = cbf_avoid_colli.cbf_mov_static_obs:main',
            'cbf_scene_rg = cbf_avoid_colli.cbf_scene_rg:main', 
            'cbf_scene_exp = cbf_avoid_colli.cbf_scene_exp:main'
        ],
    },
)
