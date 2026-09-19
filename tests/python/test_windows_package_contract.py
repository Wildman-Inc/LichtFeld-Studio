# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise CPack and the Windows ZIP contract without compiling the app."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CMAKE = shutil.which("cmake")
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


@unittest.skipUnless(os.name == "nt" and CMAKE and POWERSHELL and shutil.which("ninja"),
                     "Windows, CMake, Ninja and PowerShell are required")
class WindowsPackageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.sdk = cls.root / "sdk"
        files = {
            "sdk/bin/amdhip64_7.dll": b"selected runtime",
            "sdk/.kpack/rand_lib_gfx1151.kpack": b"selected device kernels",
            "sdk/share/doc/amd_comgr/LICENSE.txt": b"selected SDK notice",
            "sdk/share/therock/therock_manifest.json": b'{"the_rock_commit":"current"}',
            "sdk/.info/version": b"10.1.0\n",
            "LichtFeld-Studio.exe": b"fixture executable",
            "msvcp140.dll": b"fixture MSVC runtime",
            "vcruntime140.dll": b"fixture MSVC runtime",
            "LICENSE": b"fixture Studio license",
            "THIRD_PARTY_LICENSES.md": b"fixture third-party notice",
        }
        for name, data in files.items():
            path = cls.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        cmake_source = (PROJECT_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        cpack_block = cmake_source[cmake_source.index(
            "# The Windows one-shot script packages the same portable install tree as CI."):]
        (cls.root / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.24)\n'
            'project(PackageFixture VERSION 0.5.3 LANGUAGES NONE)\n'
            'set(CMAKE_INSTALL_BINDIR bin)\nset(BUILD_PORTABLE ON)\nset(USE_HIP ON)\n'
            f'include("{PROJECT_ROOT.as_posix()}/cmake/WindowsROCm.cmake")\n'
            f'lfs_windows_rocm_install_artifacts("{cls.sdk.as_posix()}" '
            f'"{cls.sdk.as_posix()}/bin/amdhip64_7.dll" '
            f'"{cls.sdk.as_posix()}/.kpack/rand_lib_gfx1151.kpack")\n'
            'install(FILES LichtFeld-Studio.exe msvcp140.dll vcruntime140.dll DESTINATION bin)\n'
            'install(FILES LICENSE DESTINATION .)\n'
            'install(FILES THIRD_PARTY_LICENSES.md DESTINATION licenses)\n' + cpack_block,
            encoding="utf-8",
        )
        cls.build = cls.root / "build"
        for command in (
            [CMAKE, "-S", str(cls.root), "-B", str(cls.build), "-G", "Ninja"],
            [CMAKE, "--build", str(cls.build), "--target", "package", "--parallel", "32"],
        ):
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)
        cls.archive = cls.build / "LichtFeld-Studio-for-ROCm-0.5.3-windows-x64.zip"
        cls.manifest = cls.build / "rocm-artifact-sha256-manifest.txt"
        cls.runner = cls.root / "validate.ps1"
        cls.runner.write_text(
            "param([string]$ArchivePath, [string]$ManifestPath, [string]$ChecksumPath)\n"
            "$ErrorActionPreference = 'Stop'\n"
            "$tokens = $null; $parseErrors = $null\n"
            "$ast = [System.Management.Automation.Language.Parser]::ParseFile(\n"
            f"    '{PROJECT_ROOT.as_posix()}/build_lichtfeld.ps1', [ref]$tokens, [ref]$parseErrors)\n"
            "if ($parseErrors) { throw ($parseErrors | Out-String) }\n"
            "$ast.FindAll({ param($node)\n"
            "    $node -is [System.Management.Automation.Language.FunctionDefinitionAst]\n"
            "}, $false) | ForEach-Object { . ([scriptblock]::Create($_.Extent.Text)) }\n"
            "Test-PackageArchiveContract $ArchivePath HIP $ManifestPath\n"
            "if ($ChecksumPath) { Test-PackageChecksum $ArchivePath $ChecksumPath }\n",
            encoding="utf-8",
        )

    def validate(self, archive, *, success, checksum=""):
        result = subprocess.run(
            [POWERSHELL, "-NoProfile", "-File", str(self.runner), str(archive),
             str(self.manifest), str(checksum)], capture_output=True, text=True,
        )
        if success:
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        else:
            self.assertNotEqual(0, result.returncode, result.stdout + result.stderr)

    def altered_archive(self, *, replace=None, omit=None):
        output = self.root / f"{self._testMethodName}.zip"
        replacements = dict(replace or {})
        with zipfile.ZipFile(self.archive) as source, zipfile.ZipFile(output, "w") as target:
            for entry in source.infolist():
                if entry.filename != omit and entry.filename not in replacements:
                    target.writestr(entry, source.read(entry))
            for name, content in replacements.items():
                target.writestr(name, content)
        return output

    def test_cpack_layout_and_checksum_match_selected_sdk(self):
        self.validate(self.archive, success=True, checksum=f"{self.archive}.sha256")
        with zipfile.ZipFile(self.archive) as archive:
            self.assertIn(".kpack/rand_lib_gfx1151.kpack", archive.namelist())
            self.assertEqual(b"10.1.0\n", archive.read(
                "licenses/ROCm/provenance/rocm-sdk-version.txt"))

    def test_changed_runtime_is_rejected(self):
        self.validate(self.altered_archive(replace={"bin/amdhip64_7.dll": b"other SDK"}), success=False)

    def test_missing_device_archive_is_rejected(self):
        self.validate(self.altered_archive(omit=".kpack/rand_lib_gfx1151.kpack"), success=False)

    def test_stale_sdk_files_are_rejected(self):
        for name in ("bin/hiprtc_old.dll", "licenses/ROCm/runtime/stale.txt"):
            with self.subTest(name=name):
                self.validate(self.altered_archive(replace={name: b"stale SDK"}), success=False)

    def test_changed_packaged_manifest_is_rejected(self):
        self.validate(self.altered_archive(replace={
            "licenses/ROCm/artifact-sha256-manifest.txt": b"wrong manifest"}), success=False)

    def test_system_test_and_unsafe_paths_are_rejected(self):
        for name in ("bin/kernel32.dll", "bin/gtest.dll", "../outside.txt"):
            with self.subTest(name=name):
                self.validate(self.altered_archive(replace={name: b"unexpected"}), success=False)


if __name__ == "__main__":
    unittest.main()
