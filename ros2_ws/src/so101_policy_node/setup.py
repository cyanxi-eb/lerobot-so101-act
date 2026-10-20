from setuptools import setup

package_name = "so101_policy_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/policy_node.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="chenyanxi",
    maintainer_email="cyanxi-eb@users.noreply.github.com",
    description="ROS2 deployment skeleton for the SO-101 imitation-learning policies",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "policy_node = so101_policy_node.policy_node:main",
        ],
    },
)
