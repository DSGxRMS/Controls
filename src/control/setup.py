from setuptools import setup, find_packages

package_name = 'control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'run_control = control.control_loop:main',
            'test_control = control.test_control:main',
            'skidpad_path = control.path_planner_skidpad:main',
            'controls_final = control.controls_final:main',
        ],
    },
)
