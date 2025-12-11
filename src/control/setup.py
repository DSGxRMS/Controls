from setuptools import setup

package_name = 'control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='armaanm',
    maintainer_email='armaanmahajanbg@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pp_node = control.pp_node:main',
            'run_lqr = control.lqr:main',
            'run_ppc = control.ppc:main',
            'run_stanley = control.stanley:main',
            'vel_profiler = control.velocity_profiler:main',
            'control_plot = control.control_plot:main',
        ],
    },
)
