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


class LichtFeldMcpBridgeInitializationTests(unittest.TestCase):
    def setUp(self) -> None:
        for name, value in (
            ("_native_initialized", False),
            ("_attached_to_existing_instance", False),
        ):
            state_patch = patch.object(bridge, name, value)
            state_patch.start()
            self.addCleanup(state_patch.stop)
        for name in ("log", "install_signal_handlers", "cleanup_spawned_processes"):
            function_patch = patch.object(bridge, name)
            function_patch.start()
            self.addCleanup(function_patch.stop)

    def test_cached_discovery_does_not_contact_or_start_native_server(self) -> None:
        initialize = {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        }
        ping = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
        tools = {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}
        manifest = {"tools": [{"name": "scene_state", "inputSchema": {"type": "object"}}]}
        messages = [
            initialize,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            ping,
            tools,
            None,
        ]
        with (
            patch.object(bridge, "read_message", side_effect=messages),
            patch.object(bridge, "write_message") as write,
            patch.object(bridge, "read_tools_cache", return_value=manifest),
            patch.object(bridge, "post_json") as post,
            patch.object(bridge, "ensure_server_ready") as startup,
        ):
            self.assertEqual(bridge.main(), 0)

        post.assert_not_called()
        startup.assert_not_called()
        self.assertFalse(bridge._native_initialized)
        self.assertEqual(
            [call.args[0] for call in write.call_args_list],
            [
                bridge.initialize_response(initialize),
                bridge.success_response(ping, {}),
                bridge.success_response(tools, manifest),
            ],
        )

    def test_native_handshake_precedes_live_requests_and_is_reused(self) -> None:
        resource = {
            "jsonrpc": "2.0", "id": 8, "method": "resources/read",
            "params": {"uri": "lichtfeld://runtime/state"},
        }
        tools = {"jsonrpc": "2.0", "id": 9, "method": "tools/list"}
        initialized = False
        notified = False

        def native(payload, timeout_s):
            nonlocal initialized, notified
            if payload["method"] == "initialize":
                self.assertFalse(initialized)
                initialized = True
                self.assertEqual(payload["params"]["capabilities"], {})
                self.assertEqual(payload["params"]["clientInfo"]["name"], bridge.BRIDGE_NAME)
                return bridge.success_response(payload, {"protocolVersion": bridge.DEFAULT_PROTOCOL_VERSION})
            if payload["method"] == "notifications/initialized":
                self.assertTrue(initialized)
                self.assertNotIn("id", payload)
                notified = True
                return None
            self.assertTrue(initialized and notified)
            return bridge.success_response(payload, {"method": payload["method"]})

        with patch.object(bridge, "post_json", side_effect=native) as post:
            resource_response = bridge.forward_message(resource)
            tools_response = bridge.forward_message(tools)

        self.assertEqual(resource_response["id"], resource["id"])
        self.assertEqual(tools_response["id"], tools["id"])
        self.assertEqual(
            [call.args[0]["method"] for call in post.call_args_list],
            ["initialize", "notifications/initialized", "resources/read", "tools/list"],
        )
        self.assertTrue(bridge._native_initialized)

    def test_failed_native_handshake_does_not_dispatch_request(self) -> None:
        rejection = {"error": {"code": -32602, "message": "Unsupported protocol"}}
        request = {"jsonrpc": "2.0", "id": 10, "method": "tools/call"}
        with patch.object(bridge, "post_json", return_value=rejection) as post:
            with self.assertRaisesRegex(RuntimeError, "Native MCP initialization failed"):
                bridge.forward_message(request)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.args[0]["method"], "initialize")
        self.assertFalse(bridge._native_initialized)

    def test_unavailable_endpoint_starts_then_initializes_before_dispatch(self) -> None:
        request = {"jsonrpc": "2.0", "id": 11, "method": "resources/list"}
        expected = bridge.success_response(request, {"resources": []})
        with (
            patch.object(bridge, "read_message", side_effect=[request, None]),
            patch.object(bridge, "write_message") as write,
            patch.object(bridge, "ensure_server_ready") as startup,
            patch.object(
                bridge, "post_json",
                side_effect=[OSError("offline"), {"result": {}}, None, expected],
            ) as post,
        ):
            self.assertEqual(bridge.main(), 0)

        startup.assert_called_once_with()
        write.assert_called_once_with(expected)
        self.assertEqual(
            [call.args[0]["method"] for call in post.call_args_list],
            ["initialize", "initialize", "notifications/initialized", "resources/list"],
        )

    def test_native_restart_reinitializes_only_after_explicit_rejection(self) -> None:
        bridge._native_initialized = True
        request = {
            "jsonrpc": "2.0", "id": 12, "method": "tools/call",
            "params": {"name": "training_start", "arguments": {}},
        }
        rejection = {
            "jsonrpc": "2.0", "id": request["id"],
            "error": {
                "code": -32600,
                "message": "Server not initialized. Call 'initialize' first.",
            },
        }
        expected = bridge.success_response(request, {"content": []})
        with patch.object(
            bridge, "post_json", side_effect=[rejection, {"result": {}}, None, expected],
        ) as post:
            self.assertEqual(bridge.forward_message(request), expected)

        self.assertEqual(
            [call.args[0]["method"] for call in post.call_args_list],
            ["tools/call", "initialize", "notifications/initialized", "tools/call"],
        )
        self.assertEqual(post.call_args_list[0].args[0], post.call_args_list[-1].args[0])

    def test_other_native_errors_are_not_retried(self) -> None:
        bridge._native_initialized = True
        request = {"jsonrpc": "2.0", "id": 13, "method": "tools/call"}
        rejection = {"error": {"code": -32600, "message": "Invalid tool request"}}
        with patch.object(bridge, "post_json", return_value=rejection) as post:
            self.assertIs(bridge.forward_message(request), rejection)
        post.assert_called_once_with(request, timeout_s=30.0)
        self.assertTrue(bridge._native_initialized)

    def test_transport_failure_invalidates_native_handshake(self) -> None:
        bridge._native_initialized = True
        request = {"jsonrpc": "2.0", "id": 14, "method": "resources/list"}
        with patch.object(bridge, "post_json", side_effect=OSError("disconnected")):
            with self.assertRaisesRegex(OSError, "disconnected"):
                bridge.forward_message(request)
        self.assertFalse(bridge._native_initialized)

    def test_notification_accepts_empty_http_response(self) -> None:
        with patch.object(bridge.urllib.request, "urlopen") as urlopen:
            response = urlopen.return_value.__enter__.return_value
            response.read.return_value = b""
            self.assertIsNone(bridge.post_json(
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
            ))


if __name__ == "__main__":
    unittest.main()
