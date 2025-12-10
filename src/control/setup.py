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
            'run_ppc = control.PPC:main',
            'run_lqr = control.LQR:main',
            'run_stanley = control.Stanley:main',
            'run_velprofile = control.Vel_pro:main'
            'control_plotter = control.realtime_plot:main',
            'path_planner = control.path_planner_node:main',
        ],
    },
)
