from setuptools import setup

package_name = 'control_v2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['resource/boa_constrictor.csv']),
        ('share/' + package_name, ['resource/small_track.csv']),
        ('share/' + package_name, ['resource/pathpoints_shifted.csv'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='omnaphade199@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ctrl_v2_test = control_v2.test_control:main',
            'pp_publisher = control_v2.pp_publisher:main',
            'control_loop = control_v2.control_loop:main',
            'velocity_profiler = control_v2.Vel_pro:main',
        ],
    },
)
