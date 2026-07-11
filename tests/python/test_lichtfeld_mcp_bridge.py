from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


BRIDGE_PATH = Path(__file__).parents[2] / "scripts" / "lichtfeld_mcp_bridge.py"
LAUNCHER_PATH = BRIDGE_PATH.with_name("lichtfeld_mcp_bridge_launcher.cmake")
SPEC = importlib.util.spec_from_file_location("lichtfeld_mcp_bridge", BRIDGE_PATH)
assert SPEC is not None and SPEC.loader is not None
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def _executable_name() -> str:
    return "LichtFeld-Studio.exe" if os.name == "nt" else "LichtFeld-Studio"


class LichtFeldMcpBridgeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cmake"), "CMake is required for the MCP launcher")
    def test_cmake_launcher_selects_a_working_python(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe_script = Path(temp_dir) / "probe.py"
            probe_script.write_text('print("LFS_CMAKE_LAUNCHER_OK")\n', encoding="utf-8")

            result = subprocess.run(
                [
                    "cmake",
                    f"-DLICHTFELD_MCP_BRIDGE_SCRIPT={probe_script.as_posix()}",
                    "-P",
                    str(LAUNCHER_PATH),
                ],
                cwd=BRIDGE_PATH.parents[1],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "LFS_CMAKE_LAUNCHER_OK")

    def test_build_hip_precedes_generic_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            hip_executable = root / "build-hip" / _executable_name()
            generic_executable = root / "build" / _executable_name()
            hip_executable.parent.mkdir()
            generic_executable.parent.mkdir()
            hip_executable.touch()
            generic_executable.touch()

            with patch.dict(os.environ, {}, clear=True), patch.object(bridge, "REPO_ROOT", root):
                candidates = bridge.executable_candidates()

            self.assertEqual(
                candidates[:2],
                [hip_executable.resolve(), generic_executable.resolve()],
            )

    def test_visual_studio_release_output_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            release_executable = root / "build" / "Release" / _executable_name()
            release_executable.parent.mkdir(parents=True)
            release_executable.touch()

            with patch.dict(os.environ, {}, clear=True), patch.object(bridge, "REPO_ROOT", root):
                candidates = bridge.executable_candidates()

            self.assertEqual(candidates, [release_executable.resolve()])

    def test_linux_rocm_build_roots_are_discovered_in_priority_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rocm_executable = root / "build-rocm" / _executable_name()
            multi_executable = root / "build-rocm-multi" / _executable_name()
            rocm_executable.parent.mkdir(parents=True)
            multi_executable.parent.mkdir(parents=True)
            rocm_executable.touch()
            multi_executable.touch()

            with patch.dict(os.environ, {}, clear=True), patch.object(bridge, "REPO_ROOT", root):
                candidates = bridge.executable_candidates()

            self.assertEqual(
                candidates,
                [rocm_executable.resolve(), multi_executable.resolve()],
            )

    def test_launch_command_does_not_inject_optional_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            executable = Path(temp_dir) / _executable_name()
            executable.touch()

            with patch.object(bridge, "executable_candidates", return_value=[executable]):
                command = bridge.pick_launch_command()

            self.assertEqual(command, [str(executable)])


if __name__ == "__main__":
    unittest.main()
