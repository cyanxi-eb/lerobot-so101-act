"""Skeleton checks for the (unexecuted) ROS2 inference node.

The ROS2 package is not run in this project: there is no ROS2 installation on
the development machine.  These tests only prove that the node source is
valid Python and that the file documents its import guard.
"""

from __future__ import annotations

import py_compile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NODE = REPO_ROOT / "ros2_ws" / "src" / "so101_policy_node" / "so101_policy_node" / "policy_node.py"
PACKAGE = NODE.parents[1]


def test_node_source_compiles(tmp_path):
    assert NODE.exists(), NODE
    target = tmp_path / "policy_node.pyc"
    py_compile.compile(str(NODE), cfile=str(target), doraise=True)
    assert target.exists()


def test_node_has_import_guard():
    source = NODE.read_text(encoding="utf-8")
    assert "import rclpy" in source
    assert "ImportError" in source
    assert "not installed" in source


def test_package_manifest_and_entrypoint():
    package_xml = PACKAGE / "package.xml"
    setup_py = PACKAGE / "setup.py"
    assert package_xml.exists() and setup_py.exists()
    xml = package_xml.read_text(encoding="utf-8")
    assert "<name>so101_policy_node</name>" in xml
    setup = setup_py.read_text(encoding="utf-8")
    assert "so101_policy_node.policy_node:main" in setup
    assert (PACKAGE / "resource" / "so101_policy_node").exists()
