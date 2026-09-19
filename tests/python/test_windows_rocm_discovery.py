# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise SDK discovery against split Windows ROCm wheel layouts."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CMAKE = shutil.which("cmake")


@unittest.skipUnless(CMAKE, "CMake is required for SDK discovery tests")
class WindowsRocmDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.core = self.root / "_rocm_sdk_core"
        self.devel = self.root / "_rocm_sdk_devel"

    def touch(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def run_cmake(self, script, *, env=None, success=True):
        path = self.root / "check.cmake"
        path.write_text(
            "cmake_minimum_required(VERSION 3.24)\n"
            f'include("{PROJECT_ROOT.as_posix()}/cmake/WindowsROCm.cmake")\n'
            + script,
            encoding="utf-8",
        )
        environment = {
            key: value
            for key, value in os.environ.items()
            if "ROCM" not in key.upper() and not key.upper().startswith("HIP_")
            and key.upper() not in {name.upper() for name in (env or {})}
        }
        environment.update(env or {})
        result = subprocess.run(
            [CMAKE, "-P", str(path)], capture_output=True, text=True, env=environment
        )
        if success:
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        else:
            self.assertNotEqual(0, result.returncode)
        return result

    def make_crt(self, redist):
        crt = redist / "x64/Microsoft.VC145.CRT"
        for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll",
                     "msvcp140_atomic_wait.dll", "msvcp140d.dll"):
            self.touch(crt / name)
        return crt

    def assert_crt(self, expected, *, includes="", explicit="", program_files=""):
        self.run_cmake(
            f'set(ENV{{ProgramFiles}} "{program_files}")\n'
            f'set(ENV{{VCToolsRedistDir}} "{explicit}")\n'
            f'set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "{includes}")\n'
            'lfs_windows_rocm_find_msvc_runtime(actual)\n'
            f'foreach(file IN LISTS actual)\n'
            f'  if(NOT file MATCHES "^{expected.as_posix()}/" OR file MATCHES "140d")\n'
            '    message(FATAL_ERROR "Wrong or debug CRT: ${actual}")\n'
            '  endif()\n'
            'endforeach()\n'
            'list(LENGTH actual count)\n'
            'if(NOT count EQUAL 4 OR MSVC)\n'
            '  message(FATAL_ERROR "Incomplete CRT or changed frontend flag")\n'
            'endif()\n',
            env={"VCToolsRedistDir": explicit, "ProgramFiles": program_files},
        )

    def test_explicit_msvc_redist_precedes_compiler_headers(self):
        explicit = self.root / "explicit/14.44.1"
        expected = self.make_crt(explicit)
        vs = self.root / "VS18/Insiders"
        self.make_crt(vs / "VC/Redist/MSVC/14.50.1")
        self.assert_crt(expected, explicit=explicit.as_posix(),
                        includes=f"{vs.as_posix()}/VC/Tools/MSVC/14.50.9/include")

    def test_msvc_redist_follows_compiler_vs_and_skips_older_runtime(self):
        vs = self.root / "VS18/Insiders"
        expected = self.make_crt(vs / "VC/Redist/MSVC/14.50.1")
        self.make_crt(vs / "VC/Redist/MSVC/14.44.1")
        installed = self.root / "Program Files"
        self.make_crt(installed / "Microsoft Visual Studio/2022/Community/VC/Redist/MSVC/14.44.9")
        self.assert_crt(expected, program_files=installed.as_posix(),
                        includes=f"{vs.as_posix()}/VC/Tools/MSVC/14.50.9/include")

    def test_msvc_redist_fallback_sorts_runtime_version_not_vs_name(self):
        installed = self.root / "Program Files"
        expected = self.make_crt(installed / "Microsoft Visual Studio/18/Insiders/VC/Redist/MSVC/14.50.1")
        self.make_crt(installed / "Microsoft Visual Studio/2022/Community/VC/Redist/MSVC/14.44.9")
        self.assert_crt(expected, program_files=installed.as_posix())

    def test_selected_core_keeps_sibling_headers_despite_other_python(self):
        self.touch(self.devel / "include/hipcub/hipcub.hpp")
        other = self.root / "other_python"
        wrong_devel = other / "_rocm_sdk_devel"
        self.touch(wrong_devel / "include/hipcub/hipcub.hpp")
        self.touch(wrong_devel / "__init__.py")
        self.touch(other / "rocm_sdk/__init__.py")
        (other / "rocm_sdk/_devel.py").write_text(
            f"def get_devel_root(): return {wrong_devel.as_posix()!r}\n",
            encoding="utf-8",
        )
        self.run_cmake(
            f'set(_lfs_python_for_rocm_devel "{Path(sys.executable).as_posix()}")\n'
            f'lfs_windows_rocm_find_devel_root("{self.core.as_posix()}" actual)\n'
            f'if(NOT actual STREQUAL "{self.devel.as_posix()}")\n'
            '  message(FATAL_ERROR "Mixed SDK headers: ${actual}")\n'
            "endif()\n",
            env={"PYTHONPATH": str(other)},
        )

    def test_multi_arch_runtime_excludes_stale_device_wheel(self):
        runtime = self.root / "_rocm_sdk_libraries/bin"
        stale = self.root / "_rocm_sdk_libraries_gfx1151/bin"
        self.touch(self.core / "bin/amdhip64_7.dll")
        self.touch(runtime / "hiprand.dll")
        self.touch(stale / "hiprand.dll")
        self.run_cmake(
            f'lfs_windows_rocm_find_runtime_dirs("{self.core.as_posix()}" actual)\n'
            f'set(expected "{self.core.as_posix()}/bin;{runtime.as_posix()}")\n'
            'if(NOT actual STREQUAL expected)\n'
            '  message(FATAL_ERROR "Mixed runtime wheels: ${actual}")\n'
            "endif()\n"
        )

    def test_legacy_runtime_wheel_remains_supported(self):
        runtime = self.root / "_rocm_sdk_libraries_gfx1151/bin"
        self.touch(runtime / "hiprand.dll")
        self.run_cmake(
            f'lfs_windows_rocm_find_runtime_dirs("{self.core.as_posix()}" actual)\n'
            f'if(NOT actual STREQUAL "{runtime.as_posix()}")\n'
            '  message(FATAL_ERROR "Missing legacy runtime: ${actual}")\n'
            "endif()\n"
        )

    def test_explicit_sdk_cannot_fall_back_to_another_installation(self):
        fallback = self.root / "fallback"
        self.touch(fallback / "include/hip/hip_runtime.h")
        result = self.run_cmake(
            f'set(LFS_ROCM_PATH "{self.core.as_posix()}")\n'
            f'set(ROCM_PATH "{fallback.as_posix()}")\n'
            "lfs_windows_rocm_find_root(actual)\n",
            success=False,
        )
        self.assertIn("LFS_ROCM_PATH does not contain a ROCm/HIP SDK", result.stderr)

    def test_explicit_environment_sdk_cannot_fall_back_to_another_installation(self):
        fallback = self.root / "fallback"
        self.touch(fallback / "include/hip/hip_runtime.h")
        result = self.run_cmake(
            "lfs_windows_rocm_find_root(actual)\n",
            env={"LFS_ROCM_PATH": str(self.core), "ROCM_PATH": str(fallback)},
            success=False,
        )
        self.assertIn("LFS_ROCM_PATH does not contain a ROCm/HIP SDK", result.stderr)

    def test_versioned_runtime_is_recognized(self):
        self.touch(self.core / "bin/amdhip64_7.dll")
        self.run_cmake(
            f'lfs_windows_rocm_root_is_usable("{self.core.as_posix()}" actual)\n'
            "if(NOT actual)\n"
            '  message(FATAL_ERROR "Versioned HIP runtime was not recognized")\n'
            "endif()\n"
        )

    @unittest.skipUnless(shutil.which("ninja"), "Ninja is required for runtime staging graph checks")
    def test_runtime_staging_configures_with_both_output_layouts(self):
        cmake_source = (PROJECT_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        staging = cmake_source.split("# The split Windows ROCm wheels", 1)[1]
        staging = "# The split Windows ROCm wheels" + staging.split(
            "# =============================================================================", 1
        )[0]
        self.touch(self.core / "bin/amdhip64_7.dll")
        source = self.root / "source"
        module_source = source / "src/python"
        module_source.mkdir(parents=True)
        (module_source / "CMakeLists.txt").write_text(
            "add_library(lfs_py MODULE IMPORTED GLOBAL)\n"
            'set_target_properties(lfs_py PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/${layout}lichtfeld.pyd")\n',
            encoding="utf-8",
        )
        (source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.30)\nproject(StagingGraph LANGUAGES NONE)\n"
            "set(WIN32 TRUE)\nset(USE_HIP TRUE)\n"
            f'set(LFS_ROCM_PATH "{self.core.as_posix()}")\n'
            "function(lfs_windows_rocm_collect_runtime_files root output)\n"
            '  set(${output} "${root}/bin/amdhip64_7.dll" PARENT_SCOPE)\nendfunction()\n'
            "function(lfs_windows_rocm_collect_runtime_kpacks root architectures runtime_files output)\n"
            '  set(${output} "${MOCK_KPACK}" PARENT_SCOPE)\nendfunction()\n'
            "get_property(multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)\n"
            'if(multi)\n  set(layout "Release/")\nendif()\n'
            "add_executable(StagingGraph IMPORTED GLOBAL)\n"
            'set_target_properties(StagingGraph PROPERTIES IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/${layout}Studio.exe")\n'
            "add_subdirectory(src/python)\n" + staging,
            encoding="utf-8",
        )
        for generator in ("Ninja", "Ninja Multi-Config"):
            with self.subTest(generator=generator):
                result = subprocess.run(
                    [CMAKE, "-S", str(source), "-B", str(self.root / generator), "-G", generator],
                    capture_output=True, text=True,
                )
                self.assertEqual(0, result.returncode, result.stdout + result.stderr)

        asset = self.root / "rand_lib_gfx1151.kpack"
        asset.write_bytes(b"device archive fixture")
        for generator in ("Ninja", "Ninja Multi-Config"):
            with self.subTest(generator=generator, external_device_archive=True):
                build = self.root / generator
                result = subprocess.run(
                    [CMAKE, "-S", str(source), "-B", str(build), "-G", generator,
                     f"-DMOCK_KPACK={asset.as_posix()}"],
                    capture_output=True, text=True,
                )
                if generator == "Ninja":
                    self.assertNotEqual(0, result.returncode)
                    self.assertIn("Ninja Multi-Config", " ".join(result.stderr.split()))
                    self.assertFalse((self.root / ".kpack").exists())
                else:
                    self.assertEqual(0, result.returncode, result.stdout + result.stderr)
                    staged = subprocess.run(
                        [CMAKE, "-P", str(build / "stage_rocm_kpack_Release.cmake")],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(0, staged.returncode, staged.stdout + staged.stderr)
                    for relative in (".kpack", "src/.kpack", "src/python/.kpack"):
                        self.assertEqual(asset.read_bytes(), (build / relative / asset.name).read_bytes())

    @unittest.skipUnless(os.environ.get("LFS_TEST_ROCM_SDK"), "Set LFS_TEST_ROCM_SDK for installed SDK inspection")
    def test_installed_runtime_closure_does_not_bundle_unused_math_libraries(self):
        sdk = Path(os.environ["LFS_TEST_ROCM_SDK"]).resolve()
        stale_dll = self.root / "another_sdk/old.dll"
        self.touch(stale_dll)
        result = self.run_cmake(
            f'set(_dependency "{stale_dll.as_posix()}" CACHE FILEPATH "stale find_file result")\n'
            f'lfs_windows_rocm_collect_runtime_files("{sdk.as_posix()}" actual)\n'
            f'lfs_windows_rocm_collect_runtime_kpacks("{sdk.as_posix()}" "gfx1151" "${{actual}}" assets)\n'
            'message(STATUS "runtime_files=${actual}")\n'
            'message(STATUS "runtime_assets=${assets}")\n'
        )
        runtime_line = next(
            line.split("runtime_files=", 1)[1]
            for line in result.stdout.splitlines()
            if "runtime_files=" in line
        )
        paths = [Path(path) for path in runtime_line.split(";")]
        names = {path.name.lower() for path in paths}
        self.assertTrue(any(name.startswith("amdhip64") for name in names))
        self.assertIn("hiprand.dll", names)
        self.assertIn("rocrand.dll", names)
        self.assertIn("amd_comgr.dll", names)
        self.assertFalse(names & {"rocblas.dll", "hipblas.dll", "miopen.dll"})
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertTrue(all(path.is_relative_to(sdk.parent) for path in paths))
        assets = next(
            line.split("runtime_assets=", 1)[1]
            for line in result.stdout.splitlines()
            if "runtime_assets=" in line
        )
        self.assertEqual(
            sdk.parent / "_rocm_sdk_libraries/.kpack/rand_lib_gfx1151.kpack", Path(assets)
        )


if __name__ == "__main__":
    unittest.main()
