# LichtFeld-Studio One-Shot Build Script for Windows
# This script verifies prerequisites, sets up dependencies, and builds the project
# Usage: .\build_lichtfeld.ps1 [-GpuBackend CUDA|HIP] [-Configuration Debug|Release] [-Package] [-Clean] [-Help]

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release',

    [ValidateSet('CUDA', 'HIP')]
    [string]$GpuBackend = 'CUDA',

    [string]$RocmPath = '',

    [string]$AmdgpuArch = 'gfx1151',

    [switch]$SkipVerification,
    [switch]$SkipVcpkg,
    [switch]$SkipLibTorch,
    [switch]$Package,
    [switch]$Clean,
    [switch]$Help
)

if ($Help) {
    Write-Host @"
LichtFeld Studio One-Shot Build Script

Usage: .\build_lichtfeld.ps1 [options]

This script automatically:
  1. Verifies build prerequisites for the selected GPU backend
  2. Sets up vcpkg in the parent directory
  3. Downloads CUDA LibTorch (Debug & Release) for CUDA builds
  4. Initializes git submodules
  5. Configures and builds LichtFeld Studio
  6. Optionally creates a portable ZIP package and SHA-256 checksum

Options:
  -Configuration <Debug|Release>  Build configuration (default: Release)
  -GpuBackend <CUDA|HIP>           GPU backend (default: CUDA)
  -RocmPath <path>                 ROCm/HIP SDK root for HIP builds
  -AmdgpuArch <arch[;arch...]>     AMDGPU target(s), for example gfx1151 or gfx90a;gfx942
  -SkipVerification               Skip environment verification
  -SkipVcpkg                      Skip vcpkg setup
  -SkipLibTorch                   Skip CUDA LibTorch download
  -Package                        Create a portable ZIP package (Release only)
  -Clean                          Clean build directory before building
  -Help                           Show this help message

Examples:
  .\build_lichtfeld.ps1                            Build Release (default)
  .\build_lichtfeld.ps1 -Configuration Debug       Build Debug
  .\build_lichtfeld.ps1 -GpuBackend HIP            Build with ROCm/HIP
  .\build_lichtfeld.ps1 -GpuBackend HIP -AmdgpuArch gfx942
  .\build_lichtfeld.ps1 -Package                   Build and package Release
  .\build_lichtfeld.ps1 -Clean                     Clean and rebuild
  .\build_lichtfeld.ps1 -SkipLibTorch              Skip LibTorch download (if already present)

Notes:
  - Package builds require Configuration Release and enable BUILD_PORTABLE.
  - Windows Long Path Support: If you encounter path-too-long errors, enable long paths:
    Run as Administrator: New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
    Then restart your system.
"@
    exit 0
}

$GpuBackend = $GpuBackend.ToUpperInvariant()
if ($Package -and $Configuration -ne 'Release') {
    Write-Host "ERROR: -Package requires -Configuration Release." -ForegroundColor Red
    exit 1
}
if ($Package) {
    $Configuration = 'Release'
}

$ErrorActionPreference = 'Stop'
$ScriptPath = $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptPath
$VcpkgPath = Join-Path (Split-Path -Parent $ProjectRoot) "vcpkg"

# Track overall status
$AllChecksPassed = $true
$FailedChecks = @()

# ============================================================================
# Helper Functions
# ============================================================================

function Test-Command {
    param([string]$Command)
    try {
        if (Get-Command $Command -ErrorAction SilentlyContinue) {
            return $true
        }
    } catch {
        return $false
    }
    return $false
}

function Find-ROCmRoot {
    $Candidates = @()
    if ($RocmPath -ne "") { $Candidates += $RocmPath }
    foreach ($EnvName in @("LFS_ROCM_PATH", "ROCM_PATH", "HIP_PATH")) {
        $Value = [Environment]::GetEnvironmentVariable($EnvName)
        if ($Value -ne "") { $Candidates += $Value }
    }

    if (Test-Command "python") {
        try {
            $PythonRoot = python -c "import importlib.util, pathlib; spec = importlib.util.find_spec('_rocm_sdk_core'); print(pathlib.Path(spec.origin).resolve().parent if spec and spec.origin else '')" 2>$null
            if ($PythonRoot -ne "") { $Candidates += $PythonRoot }
        } catch {
        }
    }

    $Candidates += @(
        "C:\Program Files\AMD\ROCm\7.14.0",
        "C:\Program Files\AMD\ROCm\7.14",
        "C:\Program Files\AMD\ROCm\7.2.2",
        "C:\Program Files\AMD\ROCm\7.2.1",
        "C:\Program Files\AMD\ROCm\7.2",
        "C:\Program Files\AMD\ROCm\7.1"
    )

    foreach ($Candidate in $Candidates | Select-Object -Unique) {
        if ($Candidate -eq "") { continue }
        $Clang = Join-Path $Candidate "lib\llvm\bin\clang++.exe"
        $HipHeader = Join-Path $Candidate "include\hip\hip_runtime.h"
        $HipLib = Join-Path $Candidate "lib\amdhip64.lib"
        if ((Test-Path $Clang) -and ((Test-Path $HipHeader) -or (Test-Path $HipLib))) {
            return (Resolve-Path $Candidate).Path
        }
    }

    return $null
}

function Get-CMakeCacheValue {
    param(
        [Parameter(Mandatory = $true)][string]$Content,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $Match = [regex]::Match(
        $Content,
        "(?m)^$([regex]::Escape($Name))(?::[^=]+)?=(.+?)\r?$")
    if ($Match.Success) {
        return $Match.Groups[1].Value.Trim()
    }
    return $null
}

function Test-EquivalentPath {
    param(
        [Parameter(Mandatory = $true)][string]$Left,
        [Parameter(Mandatory = $true)][string]$Right
    )

    try {
        $NormalizedLeft = [System.IO.Path]::GetFullPath($Left).TrimEnd('\', '/')
        $NormalizedRight = [System.IO.Path]::GetFullPath($Right).TrimEnd('\', '/')
        return $NormalizedLeft.Equals(
            $NormalizedRight,
            [System.StringComparison]::OrdinalIgnoreCase)
    } catch {
        return $false
    }
}

function Remove-PackageOutputs {
    param([Parameter(Mandatory = $true)][string[]]$Paths)

    foreach ($Path in $Paths) {
        if (Test-Path -LiteralPath $Path) {
            Remove-Item -LiteralPath $Path -Force
        }
        if (Test-Path -LiteralPath $Path) {
            throw "Could not remove the previous package output: $Path"
        }
    }
}

function Get-PackageRuntimeManifestEntries {
    param([Parameter(Mandatory = $true)][string]$ManifestPath)

    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        throw "Required runtime manifest was not generated: $ManifestPath"
    }

    $Entries = @(Get-Content -LiteralPath $ManifestPath |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -ne '' })
    if ($Entries.Count -eq 0) {
        throw "Runtime manifest is empty: $ManifestPath"
    }

    $UniqueEntries = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase)
    foreach ($Entry in $Entries) {
        if ([System.IO.Path]::GetFileName($Entry) -ne $Entry -or
            $Entry.IndexOfAny([char[]]'\\/') -ge 0 -or
            -not $UniqueEntries.Add($Entry)) {
            throw "Runtime manifest contains an invalid or duplicate file name '$Entry': $ManifestPath"
        }
    }
    return $Entries
}

function Assert-ZipEntry {
    param(
        [Parameter(Mandatory = $true)]$Entries,
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )

    if (-not $Entries.Contains($Path)) {
        throw "Package ZIP is missing $Description at '$Path'."
    }
}

function Assert-ZipNoticeDirectory {
    param(
        [Parameter(Mandatory = $true)][string[]]$EntryNames,
        [Parameter(Mandatory = $true)][string]$Directory,
        [Parameter(Mandatory = $true)][string]$Description
    )

    $Prefix = "$Directory/"
    if (-not ($EntryNames | Where-Object { $_.StartsWith($Prefix, [System.StringComparison]::OrdinalIgnoreCase) } |
            Select-Object -First 1)) {
        throw "Package ZIP is missing $Description under '$Directory'."
    }
}

function Assert-ZipRuntimePolicy {
    param([Parameter(Mandatory = $true)][string[]]$EntryNames)

    $WindowsSystemDlls = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase)
    foreach ($Name in @(
        'advapi32.dll', 'bcrypt.dll', 'cfgmgr32.dll', 'comdlg32.dll',
        'combase.dll', 'crypt32.dll', 'cryptbase.dll', 'd3d12.dll', 'dbghelp.dll',
        'dnsapi.dll', 'dwmapi.dll', 'dxcore.dll', 'dxgi.dll', 'gdi32.dll',
        'imm32.dll', 'iphlpapi.dll', 'kernel32.dll', 'kernelbase.dll', 'mpr.dll',
        'mswsock.dll', 'ncrypt.dll', 'netapi32.dll', 'normaliz.dll', 'ntdll.dll',
        'ole32.dll', 'oleaut32.dll', 'powrprof.dll', 'propsys.dll', 'rpcrt4.dll',
        'sechost.dll', 'secur32.dll', 'setupapi.dll', 'shcore.dll', 'shell32.dll',
        'shlwapi.dll', 'user32.dll', 'userenv.dll', 'uxtheme.dll', 'version.dll',
        'winhttp.dll', 'wininet.dll', 'winmm.dll', 'wintrust.dll', 'ws2_32.dll',
        'win32u.dll', 'wtsapi32.dll', 'xmllite.dll')) {
        [void]$WindowsSystemDlls.Add($Name)
    }

    foreach ($EntryName in $EntryNames) {
        $LeafName = [System.IO.Path]::GetFileName($EntryName)
        if ($WindowsSystemDlls.Contains($LeafName) -or
            $LeafName -match '^(?i:api-ms-|ext-ms-).*[.]dll$') {
            throw "Package ZIP must not redistribute Windows system DLL '$EntryName'."
        }
        if ($LeafName -match '^(?i:gtest|gmock|benchmark|catch2).*[.]dll$') {
            throw "Package ZIP must not contain test-only runtime '$EntryName'."
        }
    }
}

function Test-PackageArchiveContract {
    param(
        [Parameter(Mandatory = $true)][string]$ArchivePath,
        [Parameter(Mandatory = $true)][string]$Backend,
        [string]$RuntimeManifestPath = ''
    )

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $Archive = [System.IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try {
        $EntryNames = @($Archive.Entries |
            Where-Object { -not $_.FullName.EndsWith('/') } |
            ForEach-Object { $_.FullName.Replace('\\', '/') })
        $Entries = [System.Collections.Generic.HashSet[string]]::new(
            [System.StringComparer]::OrdinalIgnoreCase)
        foreach ($EntryName in $EntryNames) {
            if (-not $Entries.Add($EntryName)) {
                throw "Package ZIP contains a duplicate entry: $EntryName"
            }
        }
        Assert-ZipRuntimePolicy $EntryNames
        $RuntimeDirectory = 'bin/'
        foreach ($RuntimePattern in @('^(?i:msvcp140).*?[.]dll$',
                '^(?i:vcruntime140).*?[.]dll$')) {
            $LocatedRuntimePattern = '^' + [regex]::Escape($RuntimeDirectory) +
                $RuntimePattern.Substring(1)
            if (-not ($EntryNames -match $LocatedRuntimePattern)) {
                throw "Package ZIP is missing a required MSVC runtime matching '$RuntimePattern'."
            }
        }

        Assert-ZipEntry $Entries 'bin/LichtFeld-Studio.exe' 'Studio executable'
        Assert-ZipEntry $Entries 'LICENSE' 'Studio license'
        Assert-ZipEntry $Entries 'licenses/THIRD_PARTY_LICENSES.md' 'Studio third-party license notice'

        if ($Backend -eq 'HIP') {
            Assert-ZipNoticeDirectory $EntryNames 'licenses/ROCm/runtime' 'a ROCm redistribution notice'
        }

        if ($Backend -eq 'HIP' -and -not $RuntimeManifestPath) {
            throw 'HIP packages require a ROCm runtime manifest.'
        }
        if ($RuntimeManifestPath) {
            $RuntimeManifestEntries = @(Get-PackageRuntimeManifestEntries $RuntimeManifestPath)
            foreach ($RuntimeEntry in $RuntimeManifestEntries) {
                Assert-ZipEntry $Entries "bin/$RuntimeEntry" "required $Backend runtime '$RuntimeEntry'"
            }
            if ($Backend -eq 'HIP') {
                $PackagedRocmRuntimes = @($EntryNames |
                    Where-Object { $_ -match '^bin/[^/]+[.]dll$' } |
                    ForEach-Object { [System.IO.Path]::GetFileName($_).ToLowerInvariant() } |
                    Where-Object { $_ -match '^(?:amd_comgr|amdhip.*|hip.*|roc.*)[.]dll$' } |
                    Sort-Object -Unique)
                $ExpectedRocmRuntimes = @($RuntimeManifestEntries |
                    ForEach-Object { $_.ToLowerInvariant() } |
                    Sort-Object -Unique)
                $RuntimeDifference = @(Compare-Object $ExpectedRocmRuntimes $PackagedRocmRuntimes)
                if ($RuntimeDifference) {
                    throw "ROCm runtime manifest does not match the package: $($RuntimeDifference -join ', ')"
                }
            }
        }
    } finally {
        $Archive.Dispose()
    }
}

function Test-PackageChecksum {
    param(
        [Parameter(Mandatory = $true)][string]$ArchivePath,
        [Parameter(Mandatory = $true)][string]$ChecksumPath
    )

    $ChecksumLine = Get-Content -LiteralPath $ChecksumPath |
        Where-Object { $_.Trim() -ne '' } |
        Select-Object -First 1
    $ChecksumMatch = [regex]::Match(
        $ChecksumLine,
        '^\s*([0-9A-Fa-f]{64})\s+(?:\*?)(.+?)\s*$')
    if (-not $ChecksumMatch.Success) {
        throw "SHA-256 sidecar is not in a supported checksum format: $ChecksumPath"
    }
    if ($ChecksumMatch.Groups[2].Value.Trim() -ne [System.IO.Path]::GetFileName($ArchivePath)) {
        throw "SHA-256 sidecar names a different archive: $ChecksumPath"
    }
    $ActualHash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash
    if (-not $ActualHash.Equals($ChecksumMatch.Groups[1].Value,
            [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "SHA-256 sidecar does not match the generated archive: $ChecksumPath"
    }
}

function Write-Status {
    param(
        [string]$Name,
        [bool]$Passed,
        [string]$Version = "",
        [string]$ErrorMessage = "",
        [string]$Solution = ""
    )

    $StatusSymbol = if ($Passed) { "[OK]" } else { "[FAIL]" }
    $StatusColor = if ($Passed) { "Green" } else { "Red" }

    Write-Host ("{0,-35}" -f $Name) -NoNewline
    Write-Host $StatusSymbol -ForegroundColor $StatusColor -NoNewline

    if ($Version -ne "") {
        Write-Host " - $Version" -ForegroundColor Gray
    } else {
        Write-Host ""
    }

    if (-not $Passed) {
        if ($ErrorMessage -ne "") {
            Write-Host "  ERROR: $ErrorMessage" -ForegroundColor Red
        }
        if ($Solution -ne "") {
            Write-Host "  SOLUTION: $Solution" -ForegroundColor Yellow
        }
        Write-Host ""
        $script:AllChecksPassed = $false
        $script:FailedChecks += $Name
    }
}

function Write-Warning-Status {
    param(
        [string]$Name,
        [string]$Message,
        [string]$Suggestion = ""
    )
    Write-Host ("{0,-35}" -f $Name) -NoNewline
    Write-Host "[WARN]" -ForegroundColor Yellow -NoNewline
    Write-Host " - $Message" -ForegroundColor Gray
    if ($Suggestion -ne "") {
        Write-Host "  SUGGESTION: $Suggestion" -ForegroundColor Cyan
    }
}

# ============================================================================
# Auto-Launch VS Developer Environment
# ============================================================================

function Test-VSDevEnvironment {
    return (($env:VSINSTALLDIR -or $env:VisualStudioVersion) -and
        $env:VisualStudioVersion -like '17.*' -and
        (Test-Command 'cl.exe'))
}

function Find-VSInstallPath {
    $VSWherePath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

    if (-not (Test-Path $VSWherePath)) {
        return $null
    }

    try {
        return & $VSWherePath -latest -products * `
            -version '[17.0,18.0)' `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath 2>$null
    } catch {
        return $null
    }
}

function Initialize-VSDevEnvironment {
    $VSInstallPath = Find-VSInstallPath

    if (-not $VSInstallPath) {
        Write-Host "ERROR: Visual Studio 2022 with C++ workload not found!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please install Visual Studio 2022 with 'Desktop development with C++' workload" -ForegroundColor Yellow
        Write-Host "Download from: https://visualstudio.microsoft.com/downloads/" -ForegroundColor Cyan
        exit 1
    }

    $VSDevShellScript = Join-Path $VSInstallPath "Common7\Tools\Launch-VsDevShell.ps1"

    if (-not (Test-Path $VSDevShellScript)) {
        Write-Host "ERROR: VS Developer PowerShell module not found at expected location" -ForegroundColor Red
        exit 1
    }

    Write-Host "Initializing the Visual Studio x64 developer environment..." -ForegroundColor Yellow
    try {
        & $VSDevShellScript -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
    } catch {
        Write-Host "ERROR: Failed to initialize the Visual Studio developer environment: $_" -ForegroundColor Red
        exit 1
    }
    if (-not (Test-Command "cl.exe")) {
        Write-Host "ERROR: Visual Studio developer environment did not expose cl.exe." -ForegroundColor Red
        exit 1
    }
    Write-Host "Visual Studio x64 developer environment initialized." -ForegroundColor Green
    Write-Host ""
}

# Check if we need to launch VS Dev environment
if (-not (Test-VSDevEnvironment)) {
    Initialize-VSDevEnvironment
}

# ============================================================================
# Environment Verification
# ============================================================================

function Test-BuildEnvironment {
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "Verifying Build Environment" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""

    $TotalChecks = 8

    # Check 1: PowerShell Version
    Write-Host "[1/$TotalChecks] Checking PowerShell..." -ForegroundColor Yellow
    $PSVer = $PSVersionTable.PSVersion
    if ($PSVer.Major -gt 5 -or ($PSVer.Major -eq 5 -and $PSVer.Minor -ge 1)) {
        Write-Status "PowerShell" $true "v$($PSVer.Major).$($PSVer.Minor)"
    } else {
        Write-Status "PowerShell" $false "v$($PSVer.Major).$($PSVer.Minor)" `
            "PowerShell 5.1 or later required" `
            "Update PowerShell: https://docs.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-windows"
    }

    # Check 2: Visual Studio / MSVC
    Write-Host "[2/$TotalChecks] Checking Visual Studio..." -ForegroundColor Yellow
    if (Test-Command "cl") {
        try {
            $ClVersion = & cl 2>&1 | Select-String "Version" | Select-Object -First 1
            Write-Status "MSVC Compiler (cl.exe)" $true $ClVersion.ToString().Trim()
        } catch {
            Write-Status "MSVC Compiler (cl.exe)" $true "Found"
        }
    } else {
        Write-Status "MSVC Compiler (cl.exe)" $false "" `
            "Visual Studio Build Tools not found" `
            "Install Visual Studio 2022 with 'Desktop development with C++'"
    }

    # Check 3: CMake
    Write-Host "[3/$TotalChecks] Checking CMake..." -ForegroundColor Yellow
    if (Test-Command "cmake") {
        try {
            $CMakeVersion = (cmake --version | Select-Object -First 1) -replace 'cmake version ', ''
            $CMakeVersionParsed = [version]($CMakeVersion -split '-')[0]

            if ($CMakeVersionParsed -ge [version]"3.30") {
                Write-Status "CMake" $true "v$CMakeVersion"
            } else {
                Write-Status "CMake" $false "v$CMakeVersion (too old)" `
                    "CMake 3.30 or later is required" `
                    "Download from: https://cmake.org/download/"
            }
        } catch {
            Write-Status "CMake" $true "Found (version check failed)"
        }
    } else {
        Write-Status "CMake" $false "" `
            "CMake not found in PATH" `
            "Download CMake 3.30+ from: https://cmake.org/download/"
    }

    # Check 4: GPU backend SDK
    if ($GpuBackend -eq 'CUDA') {
        Write-Host "[4/$TotalChecks] Checking CUDA Toolkit 12.8..." -ForegroundColor Yellow
        if (Test-Command "nvcc") {
            try {
                $NvccOutput = nvcc --version 2>&1 | Select-String "release"
                $CudaVersion = ($NvccOutput -split "release ")[1] -split "," | Select-Object -First 1

                if ($CudaVersion -match "12\.") {
                    Write-Status "CUDA Toolkit (nvcc)" $true "v$CudaVersion"
                } else {
                    Write-Status "CUDA Toolkit (nvcc)" $false "v$CudaVersion" `
                        "CUDA 12.x is required (found $CudaVersion)" `
                        "Download CUDA 12.8 from: https://developer.nvidia.com/cuda-12-8-0-download-archive"
                }
            } catch {
                Write-Status "CUDA Toolkit (nvcc)" $true "Found (version check failed)"
            }
        } else {
            Write-Status "CUDA Toolkit (nvcc)" $false "" `
                "CUDA Toolkit not found or nvcc not in PATH" `
                "Download CUDA 12.8 from: https://developer.nvidia.com/cuda-12-8-0-download-archive"
        }
    } elseif ($GpuBackend -eq 'HIP') {
        Write-Host "[4/$TotalChecks] Checking ROCm/HIP SDK..." -ForegroundColor Yellow
        $ResolvedRocm = Find-ROCmRoot
        if ($ResolvedRocm) {
            $HipVersionHeader = Join-Path $ResolvedRocm "include\hip\hip_version.h"
            $HipVersion = "Found"
            if (Test-Path $HipVersionHeader) {
                $MajorMatch = Select-String -Path $HipVersionHeader -Pattern "#define HIP_VERSION_MAJOR\s+([0-9]+)" | Select-Object -First 1
                $MinorMatch = Select-String -Path $HipVersionHeader -Pattern "#define HIP_VERSION_MINOR\s+([0-9]+)" | Select-Object -First 1
                $Major = if ($MajorMatch) { $MajorMatch.Matches.Groups[1].Value } else { "" }
                $Minor = if ($MinorMatch) { $MinorMatch.Matches.Groups[1].Value } else { "" }
                if ($Major -ne "" -and $Minor -ne "") {
                    $HipVersion = "v$Major.$Minor"
                }
            }
            Write-Status "ROCm/HIP SDK" $true "$HipVersion at $ResolvedRocm"
        } else {
            Write-Status "ROCm/HIP SDK" $false "" `
                "ROCm/HIP SDK not found" `
                "Set -RocmPath, LFS_ROCM_PATH, ROCM_PATH, HIP_PATH, or install the _rocm_sdk_core Python package"
        }
    }

    # Check 5: Git
    Write-Host "[5/$TotalChecks] Checking Git..." -ForegroundColor Yellow
    if (Test-Command "git") {
        try {
            $GitVersion = (git --version) -replace 'git version ', ''
            Write-Status "Git" $true "v$GitVersion"
        } catch {
            Write-Status "Git" $true "Found"
        }
    } else {
        Write-Status "Git" $false "" `
            "Git not found in PATH" `
            "Download Git from: https://git-scm.com/download/win"
    }

    # Check 6: Ninja
    $NinjaRequired = $GpuBackend -eq 'HIP'
    $NinjaLabel = if ($NinjaRequired) { "Checking Ninja..." } else { "Checking Ninja (optional)..." }
    Write-Host "[6/$TotalChecks] $NinjaLabel" -ForegroundColor Yellow
    if (Test-Command "ninja") {
        try {
            $NinjaVersion = ninja --version
            Write-Status "Ninja" $true "v$NinjaVersion"
        } catch {
            Write-Status "Ninja" $true "Found"
        }
    } elseif ($NinjaRequired) {
        Write-Status "Ninja" $false "" `
            "Ninja is required for $GpuBackend builds" `
            "Install via: choco install ninja OR download from https://github.com/ninja-build/ninja/releases"
    } else {
        Write-Warning-Status "Ninja" "Not found (build will use Visual Studio generator)" `
            "Install via: choco install ninja OR download from https://github.com/ninja-build/ninja/releases"
    }

    # Check 7: Disk Space
    Write-Host "[7/$TotalChecks] Checking disk space..." -ForegroundColor Yellow
    try {
        $Drive = (Get-Location).Drive
        $FreeSpaceGB = [math]::Round((Get-PSDrive $Drive.Name).Free / 1GB, 2)

        if ($FreeSpaceGB -gt 30) {
            Write-Status "Disk Space" $true "$FreeSpaceGB GB available"
        } elseif ($FreeSpaceGB -gt 15) {
            Write-Warning-Status "Disk Space" "$FreeSpaceGB GB available (may be insufficient)" `
                "Build may require 20-30 GB. Consider freeing up space."
        } else {
            Write-Status "Disk Space" $false "$FreeSpaceGB GB available" `
                "Insufficient disk space (at least 15-20 GB recommended)" `
                "Free up disk space on drive $($Drive.Name):"
        }
    } catch {
        Write-Warning-Status "Disk Space" "Could not determine free space"
    }

    # Check 8: Python
    if (Test-Command "python") {
        Write-Host "[8/$TotalChecks] Checking Python 3.12..." -ForegroundColor Yellow
        try {
            $PythonVersion = python --version 2>&1
            if ($PythonVersion -match "3\.12") {
                Write-Status "Python" $true $PythonVersion
            } else {
                Write-Warning-Status "Python" "$PythonVersion (3.12 recommended)" `
                    "Python 3.12 is recommended for best compatibility"
            }
        } catch {
            Write-Status "Python" $true "Found (version check failed)"
        }
    } else {
        Write-Host "[8/$TotalChecks] Checking Python 3.12..." -ForegroundColor Yellow
        Write-Status "Python" $false "" `
            "Python not found in PATH" `
            "Download Python 3.12 from: https://www.python.org/downloads/"
    }

    Write-Host ""

    return $script:AllChecksPassed
}

# ============================================================================
# Git Submodules Setup
# ============================================================================

function Setup-GitSubmodules {
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "Initializing Git Submodules" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""

    Push-Location $ProjectRoot
    try {
        # Check if libvterm submodule is initialized
        $LibvtermPath = Join-Path $ProjectRoot "external\libvterm"
        if (-not (Test-Path (Join-Path $LibvtermPath "src"))) {
            Write-Host "Initializing git submodules..." -ForegroundColor Yellow
            git submodule update --init --recursive
            if ($LASTEXITCODE -ne 0) {
                Write-Host "WARNING: Failed to initialize submodules" -ForegroundColor Yellow
            } else {
                Write-Host "Git submodules initialized successfully!" -ForegroundColor Green
            }
        } else {
            Write-Host "Git submodules already initialized." -ForegroundColor Green
        }
    } finally {
        Pop-Location
    }
    Write-Host ""
}

# ============================================================================
# vcpkg Setup
# ============================================================================

function Setup-Vcpkg {
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "Setting Up vcpkg" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "vcpkg location: $VcpkgPath" -ForegroundColor Gray
    Write-Host ""

    if (-not (Test-Path $VcpkgPath)) {
        Write-Host "Cloning vcpkg repository..." -ForegroundColor Yellow
        git clone https://github.com/microsoft/vcpkg.git "$VcpkgPath"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to clone vcpkg" -ForegroundColor Red
            exit 1
        }
        Write-Host "vcpkg cloned successfully!" -ForegroundColor Green
    } else {
        Write-Host "vcpkg directory exists. Updating..." -ForegroundColor Yellow
        Push-Location $VcpkgPath
        try {
            # Fetch all refs first (needed for baseline resolution)
            git fetch --all 2>&1 | Out-Null
            git pull 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Write-Host "WARNING: git pull failed, continuing with existing version" -ForegroundColor Yellow
            } else {
                Write-Host "vcpkg updated successfully!" -ForegroundColor Green
            }
        } catch {
            Write-Host "WARNING: vcpkg update failed, continuing with existing version" -ForegroundColor Yellow
        } finally {
            Pop-Location
        }
    }

    # Bootstrap vcpkg if needed
    $VcpkgExe = Join-Path $VcpkgPath "vcpkg.exe"
    if (-not (Test-Path $VcpkgExe)) {
        Write-Host ""
        Write-Host "Bootstrapping vcpkg..." -ForegroundColor Yellow
        Write-Host "This may take a few minutes..." -ForegroundColor Gray

        Push-Location $VcpkgPath
        try {
            & .\bootstrap-vcpkg.bat -disableMetrics
            if ($LASTEXITCODE -ne 0) {
                Write-Host "ERROR: Failed to bootstrap vcpkg" -ForegroundColor Red
                exit 1
            }
        } finally {
            Pop-Location
        }
        Write-Host "vcpkg bootstrapped successfully!" -ForegroundColor Green
    } else {
        Write-Host "vcpkg.exe already exists. Skipping bootstrap." -ForegroundColor Green
    }

    # Set environment variable for CMake
    $env:VCPKG_ROOT = $VcpkgPath
    Write-Host ""
    Write-Host "VCPKG_ROOT set to: $VcpkgPath" -ForegroundColor Cyan
    Write-Host ""
}

# ============================================================================
# LibTorch Download
# ============================================================================

function Setup-LibTorch {
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "Setting Up LibTorch" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""

    $ExternalDir = Join-Path $ProjectRoot "external"
    $DebugDir = Join-Path $ExternalDir "debug"
    $ReleaseDir = Join-Path $ExternalDir "release"
    $DebugLibTorch = Join-Path $DebugDir "libtorch"
    $ReleaseLibTorch = Join-Path $ReleaseDir "libtorch"

    # Create directories
    if (-not (Test-Path $ExternalDir)) {
        New-Item -ItemType Directory -Path $ExternalDir | Out-Null
        Write-Host "Created: external/" -ForegroundColor Gray
    }
    if (-not (Test-Path $DebugDir)) {
        New-Item -ItemType Directory -Path $DebugDir | Out-Null
        Write-Host "Created: external/debug/" -ForegroundColor Gray
    }
    if (-not (Test-Path $ReleaseDir)) {
        New-Item -ItemType Directory -Path $ReleaseDir | Out-Null
        Write-Host "Created: external/release/" -ForegroundColor Gray
    }

    # Download Debug LibTorch if missing
    if (-not (Test-Path $DebugLibTorch)) {
        Write-Host ""
        Write-Host "Downloading LibTorch (Debug)..." -ForegroundColor Yellow
        Write-Host "This is a large download (~3.2 GB). Please wait..." -ForegroundColor Gray

        $DebugZip = Join-Path $ProjectRoot "libtorch-debug.zip"
        $DebugUrl = "https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-debug-2.7.0%2Bcu128.zip"

        try {
            curl.exe -L -o "$DebugZip" "$DebugUrl" --progress-bar
            if ($LASTEXITCODE -ne 0) { throw "Download failed" }

            Write-Host "Extracting LibTorch (Debug)..." -ForegroundColor Yellow
            tar -xf "$DebugZip" -C "$DebugDir"
            if ($LASTEXITCODE -ne 0) { throw "Extraction failed" }

            Remove-Item $DebugZip -Force
            Write-Host "LibTorch (Debug) installed successfully!" -ForegroundColor Green
        } catch {
            Write-Host "ERROR: Failed to download/extract Debug LibTorch: $_" -ForegroundColor Red
            if (Test-Path $DebugZip) { Remove-Item $DebugZip -Force }
            exit 1
        }
    } else {
        Write-Host "LibTorch (Debug) already exists. Skipping download." -ForegroundColor Green
    }

    # Download Release LibTorch if missing
    if (-not (Test-Path $ReleaseLibTorch)) {
        Write-Host ""
        Write-Host "Downloading LibTorch (Release)..." -ForegroundColor Yellow
        Write-Host "This is a large download (~3.2 GB). Please wait..." -ForegroundColor Gray

        $ReleaseZip = Join-Path $ProjectRoot "libtorch-release.zip"
        $ReleaseUrl = "https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-2.7.0%2Bcu128.zip"

        try {
            curl.exe -L -o "$ReleaseZip" "$ReleaseUrl" --progress-bar
            if ($LASTEXITCODE -ne 0) { throw "Download failed" }

            Write-Host "Extracting LibTorch (Release)..." -ForegroundColor Yellow
            tar -xf "$ReleaseZip" -C "$ReleaseDir"
            if ($LASTEXITCODE -ne 0) { throw "Extraction failed" }

            Remove-Item $ReleaseZip -Force
            Write-Host "LibTorch (Release) installed successfully!" -ForegroundColor Green
        } catch {
            Write-Host "ERROR: Failed to download/extract Release LibTorch: $_" -ForegroundColor Red
            if (Test-Path $ReleaseZip) { Remove-Item $ReleaseZip -Force }
            exit 1
        }
    } else {
        Write-Host "LibTorch (Release) already exists. Skipping download." -ForegroundColor Green
    }

    Write-Host ""
}

# ============================================================================
# Copy DLLs to Output Directory
# ============================================================================

function Copy-RequiredDLLs {
    param([string]$BuildDir, [string]$Config)

    Write-Host ""
    Write-Host "Copying required DLLs to output directory..." -ForegroundColor Yellow

    $OutputDir = Join-Path $BuildDir $Config
    if (-not (Test-Path $OutputDir)) {
        $OutputDir = $BuildDir
    }

    if (-not (Test-Path $OutputDir)) {
        Write-Host "WARNING: Output directory not found: $OutputDir" -ForegroundColor Yellow
        Write-Host "Skipping DLL copy (CMake POST_BUILD commands should handle this)" -ForegroundColor Gray
        return
    }

    # DLLs that need to be copied from subdirectories (fallback if CMake POST_BUILD fails)
    $DLLSources = @(
        @{ Source = "src\core\event_bridge\$Config\lfs_event_bridge.dll"; Name = "lfs_event_bridge.dll" }
    )

    $DLLSources += @{ Source = "src\python\$Config\lfs_python_runtime.dll"; Name = "lfs_python_runtime.dll" }
    $DLLSources += @{ Source = "src\python\$Config\python3.dll"; Name = "python3.dll" }
    $DLLSources += @{ Source = "src\visualizer\$Config\lfs_visualizer.dll"; Name = "lfs_visualizer.dll" }
    $DLLSources += @{ Source = "src\mcp\$Config\lfs_mcp.dll"; Name = "lfs_mcp.dll" }

    $CopiedCount = 0
    foreach ($DLL in $DLLSources) {
        $SourcePath = Join-Path $BuildDir $DLL.Source
        $DestPath = Join-Path $OutputDir $DLL.Name

        if (Test-Path $SourcePath) {
            if (-not (Test-Path $DestPath)) {
                try {
                    Copy-Item $SourcePath $DestPath -Force -ErrorAction Stop
                    Write-Host "  Copied: $($DLL.Name)" -ForegroundColor Gray
                    $CopiedCount++
                } catch {
                    Write-Host "  WARNING: Failed to copy $($DLL.Name): $_" -ForegroundColor Yellow
                }
            } else {
                Write-Host "  Already exists: $($DLL.Name)" -ForegroundColor Gray
            }
        } else {
            Write-Host "  Not found (may be copied by CMake): $($DLL.Name)" -ForegroundColor Gray
        }
    }

    if ($CopiedCount -gt 0) {
        Write-Host "DLL copy completed! ($CopiedCount files)" -ForegroundColor Green
    } else {
        Write-Host "No DLLs needed copying (CMake POST_BUILD handles this)" -ForegroundColor Green
    }
}

# ============================================================================
# Build LichtFeld-Studio
# ============================================================================

function Build-LichtFeldStudio {
    $ProductDisplayName = if ($GpuBackend -eq 'HIP') {
        'LichtFeld Studio for ROCm'
    } else {
        'LichtFeld Studio'
    }
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "Building $ProductDisplayName ($Configuration)" -ForegroundColor Cyan
    Write-Host "  with Python Bindings" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""

    Push-Location $ProjectRoot
    try {
        $BuildDirName = if ($GpuBackend -eq 'HIP') {
            'build-hip'
        } else {
            'build'
        }
        $BuildDir = Join-Path $ProjectRoot $BuildDirName
        $VcpkgToolchain = Join-Path $VcpkgPath "scripts\buildsystems\vcpkg.cmake"
        $Generator = if ($GpuBackend -eq 'HIP') { "Ninja" } else { "Visual Studio 17 2022" }
        $ResolvedRocm = $null
        if ($GpuBackend -eq 'HIP') {
            $ResolvedRocm = Find-ROCmRoot
            if (-not $ResolvedRocm) {
                Write-Host "ERROR: ROCm/HIP SDK not found!" -ForegroundColor Red
                Write-Host "Set -RocmPath, LFS_ROCM_PATH, ROCM_PATH, HIP_PATH, or install the _rocm_sdk_core Python package." -ForegroundColor Yellow
                exit 1
            }
        }
        if ($Generator -eq 'Ninja' -and -not (Test-Command "ninja")) {
            Write-Host "ERROR: Ninja is required for $GpuBackend builds." -ForegroundColor Red
            Write-Host "Install Ninja and ensure ninja.exe is available on PATH." -ForegroundColor Yellow
            exit 1
        }

        # Verify vcpkg toolchain exists
        if (-not (Test-Path $VcpkgToolchain)) {
            Write-Host "ERROR: vcpkg toolchain file not found!" -ForegroundColor Red
            Write-Host "Expected: $VcpkgToolchain" -ForegroundColor Gray
            Write-Host ""
            Write-Host "Please run without -SkipVcpkg to set up vcpkg first." -ForegroundColor Yellow
            exit 1
        }

        if ($GpuBackend -eq 'CUDA') {
            # Verify CUDA LibTorch exists for the selected configuration.
            $LibTorchDebugPath = Join-Path $ProjectRoot "external\debug\libtorch"
            $LibTorchReleasePath = Join-Path $ProjectRoot "external\release\libtorch"
            $LibTorchPath = if ($Configuration -eq 'Debug') {
                $LibTorchDebugPath
            } else {
                $LibTorchReleasePath
            }

            if (-not (Test-Path $LibTorchPath)) {
                Write-Host "ERROR: LibTorch ($Configuration) not found!" -ForegroundColor Red
                Write-Host "Expected: $LibTorchPath" -ForegroundColor Gray
                Write-Host ""
                Write-Host "Please run without -SkipLibTorch to download LibTorch first." -ForegroundColor Yellow
                exit 1
            }

            # Visual Studio is multi-config, so keep both configurations available.
            if (-not (Test-Path $LibTorchDebugPath)) {
                Write-Host "WARNING: LibTorch Debug not found (multi-config generator requires both)" -ForegroundColor Yellow
                Write-Host "Expected: $LibTorchDebugPath" -ForegroundColor Gray
                Write-Host "Debug builds may fail. Run without -SkipLibTorch to download both." -ForegroundColor Yellow
                Write-Host ""
            }
            if (-not (Test-Path $LibTorchReleasePath)) {
                Write-Host "WARNING: LibTorch Release not found (multi-config generator requires both)" -ForegroundColor Yellow
                Write-Host "Expected: $LibTorchReleasePath" -ForegroundColor Gray
                Write-Host "Release builds may fail. Run without -SkipLibTorch to download both." -ForegroundColor Yellow
                Write-Host ""
            }
        }

        # Clean if requested
        if ($Clean -and (Test-Path $BuildDir)) {
            Write-Host "Cleaning build directory..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force $BuildDir
            Write-Host "Build directory cleaned." -ForegroundColor Green
            Write-Host ""
        }

        # Verify Visual Studio generator is available (only if cl.exe wasn't found earlier)
        # If we're in a VS dev environment with working cl.exe, the generator should work
        if ($Generator -like "Visual Studio*" -and -not (Test-Command "cl")) {
            $VSInstallPath = Find-VSInstallPath
            if (-not $VSInstallPath) {
                Write-Host "ERROR: Visual Studio 2022 not found!" -ForegroundColor Red
                Write-Host "The '$Generator' generator requires Visual Studio 2022 to be installed." -ForegroundColor Gray
                Write-Host ""
                Write-Host "Please install Visual Studio 2022 with 'Desktop development with C++' workload" -ForegroundColor Yellow
                Write-Host "Download from: https://visualstudio.microsoft.com/downloads/" -ForegroundColor Cyan
                exit 1
            }
        }

        # Build CMake arguments
        $CMakeArgs = @(
            "-S", ".",
            "-B", $BuildDir,
            "-G", $Generator,
            "-DCMAKE_TOOLCHAIN_FILE=$VcpkgToolchain",
            "-DLFS_GPU_BACKEND=$GpuBackend",
            "-DLFS_VCPKG_MAX_CONCURRENCY=32"
        )
        if ($Package) {
            $CMakeArgs += '-DBUILD_PORTABLE=ON'
        } else {
            # Reset cache values forced by a prior portable Studio build.
            $CMakeArgs += '-DBUILD_PORTABLE=OFF'
            if ($GpuBackend -eq 'CUDA') {
                $CMakeArgs += '-DBUILD_CUDA_PTX_ONLY=OFF'
            }
        }
        if ($Generator -like "Visual Studio*") {
            $CMakeArgs += @("-A", "x64")
        } else {
            $CMakeArgs += "-DCMAKE_BUILD_TYPE=$Configuration"
        }

        if ($GpuBackend -eq 'HIP') {
            $HipCxxCompiler = Join-Path $ResolvedRocm 'lib\llvm\bin\clang++.exe'
            $HipCCompiler = Join-Path $ResolvedRocm 'lib\llvm\bin\clang.exe'
            if (-not (Test-Path -LiteralPath $HipCxxCompiler -PathType Leaf) -or
                -not (Test-Path -LiteralPath $HipCCompiler -PathType Leaf)) {
                Write-Host "ERROR: ROCm/HIP Clang compiler pair was not found." -ForegroundColor Red
                Write-Host "Expected: $HipCxxCompiler" -ForegroundColor Gray
                Write-Host "Expected: $HipCCompiler" -ForegroundColor Gray
                exit 1
            }

            $CMakeArgs += "-DLFS_ROCM_PATH=$ResolvedRocm"
            $CMakeArgs += "-DLFS_AMDGPU_ARCH=$AmdgpuArch"
            $CMakeArgs += "-DCMAKE_CXX_COMPILER=$HipCxxCompiler"
            $CMakeArgs += "-DCMAKE_C_COMPILER=$HipCCompiler"

            $CMakeCachePath = Join-Path $BuildDir 'CMakeCache.txt'
            if (Test-Path -LiteralPath $CMakeCachePath -PathType Leaf) {
                $CMakeCacheContent = Get-Content -LiteralPath $CMakeCachePath -Raw
                $CachedCxxCompiler = Get-CMakeCacheValue $CMakeCacheContent 'CMAKE_CXX_COMPILER'
                $CachedCCompiler = Get-CMakeCacheValue $CMakeCacheContent 'CMAKE_C_COMPILER'
                $RequiresFreshConfigure =
                    -not $CachedCxxCompiler -or
                    -not $CachedCCompiler -or
                    -not (Test-EquivalentPath $CachedCxxCompiler $HipCxxCompiler) -or
                    -not (Test-EquivalentPath $CachedCCompiler $HipCCompiler)

                foreach ($CompilerToolName in @(
                    'CMAKE_CXX_COMPILER_AR',
                    'CMAKE_CXX_COMPILER_RANLIB',
                    'CMAKE_C_COMPILER_AR',
                    'CMAKE_C_COMPILER_RANLIB')) {
                    $CompilerTool = Get-CMakeCacheValue $CMakeCacheContent $CompilerToolName
                    if ($CompilerTool -and
                        [System.IO.Path]::IsPathRooted($CompilerTool) -and
                        -not (Test-Path -LiteralPath $CompilerTool -PathType Leaf)) {
                        $RequiresFreshConfigure = $true
                    }
                }

                if ($RequiresFreshConfigure) {
                    Write-Host "Refreshing CMake cache after ROCm/HIP toolchain change." -ForegroundColor Yellow
                    $CMakeArgs = @('--fresh') + $CMakeArgs
                }
            }
        }

        if ($GpuBackend -eq 'CUDA') {
            # Add LibTorch path for CUDA CMake builds when present.
            $TorchConfigPath = if ($Configuration -eq 'Debug') {
                Join-Path $ProjectRoot "external\debug\libtorch\share\cmake\Torch"
            } else {
                Join-Path $ProjectRoot "external\release\libtorch\share\cmake\Torch"
            }

            if (Test-Path $TorchConfigPath) {
                $CMakeArgs += "-DTorch_DIR=$TorchConfigPath"
            }
        }

        # Configure
        Write-Host "Configuring CMake..." -ForegroundColor Yellow
        Write-Host "  Generator: $Generator" -ForegroundColor Gray
        Write-Host "  Configuration: $Configuration" -ForegroundColor Gray
        Write-Host "  GPU Backend: $GpuBackend" -ForegroundColor Gray
        if ($Package) {
            Write-Host "  Portable Package: Enabled" -ForegroundColor Gray
        }
        if ($GpuBackend -eq 'HIP') {
            Write-Host "  ROCm/HIP SDK: $ResolvedRocm" -ForegroundColor Gray
            Write-Host "  AMDGPU targets: $AmdgpuArch" -ForegroundColor Gray
        }
        Write-Host "  Toolchain: $VcpkgToolchain" -ForegroundColor Gray
        Write-Host "  Python Bindings: Enabled" -ForegroundColor Gray
        Write-Host ""

        & cmake @CMakeArgs

        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: CMake configuration failed!" -ForegroundColor Red
            exit 1
        }

        Write-Host ""
        Write-Host "CMake configuration successful!" -ForegroundColor Green
        Write-Host ""

        # Build
        Write-Host "Building $ProductDisplayName..." -ForegroundColor Yellow
        Write-Host "This may take 10-30 minutes depending on your system..." -ForegroundColor Gray
        Write-Host ""

        if ($Generator -like "Visual Studio*") {
            $SolutionFile = Join-Path $BuildDir "LichtFeld-Studio.sln"
            msbuild $SolutionFile /p:Configuration=$Configuration /p:Platform=x64 /m:32
        } else {
            cmake --build $BuildDir --config $Configuration --parallel 32
        }

        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Build failed!" -ForegroundColor Red
            exit 1
        }

        Copy-RequiredDLLs -BuildDir $BuildDir -Config $Configuration

        $ExecutablePath = if ($Generator -like "Visual Studio*") {
            Join-Path (Join-Path $BuildDir $Configuration) "LichtFeld-Studio.exe"
        } else {
            Join-Path $BuildDir "LichtFeld-Studio.exe"
        }

        if (-not (Test-Path -LiteralPath $ExecutablePath -PathType Leaf)) {
            Write-Host "ERROR: Studio build completed without producing the expected executable." -ForegroundColor Red
            Write-Host "Expected: $ExecutablePath" -ForegroundColor Gray
            exit 1
        }
        $ExecutablePath = (Resolve-Path -LiteralPath $ExecutablePath).Path

        $PackageArchivePath = $null
        $PackageChecksumPath = $null
        if ($Package) {
            $CPackConfigPath = Join-Path $BuildDir 'CPackConfig.cmake'
            if (-not (Test-Path -LiteralPath $CPackConfigPath -PathType Leaf)) {
                Write-Host "ERROR: CPack configuration was not generated." -ForegroundColor Red
                Write-Host "Expected: $CPackConfigPath" -ForegroundColor Gray
                exit 1
            }

            $CPackConfigContent = Get-Content -LiteralPath $CPackConfigPath -Raw
            $PackageNameMatch = [regex]::Match(
                $CPackConfigContent,
                '(?m)^[ \t]*set[ \t]*\([ \t]*CPACK_PACKAGE_FILE_NAME[ \t]+"([^"]+)"[ \t]*\)[ \t]*\r?$')
            if (-not $PackageNameMatch.Success) {
                Write-Host "ERROR: CPack configuration does not define CPACK_PACKAGE_FILE_NAME." -ForegroundColor Red
                exit 1
            }
            if ($CPackConfigContent -notmatch '(?m)^[ \t]*set[ \t]*\([ \t]*CPACK_GENERATOR[ \t]+"ZIP"[ \t]*\)[ \t]*\r?$' -or
                $CPackConfigContent -notmatch '(?m)^[ \t]*set[ \t]*\([ \t]*CPACK_PACKAGE_CHECKSUM[ \t]+"SHA256"[ \t]*\)[ \t]*\r?$') {
                Write-Host "ERROR: CPack is not configured for the expected ZIP and SHA-256 outputs." -ForegroundColor Red
                exit 1
            }

            $CMakeListsPath = Join-Path $ProjectRoot 'CMakeLists.txt'
            $CMakeListsContent = Get-Content -LiteralPath $CMakeListsPath -Raw
            $ProjectVersionMatch = [regex]::Match(
                $CMakeListsContent,
                '(?m)^[ \t]*project[ \t]*\([ \t]*LichtFeld-Studio[ \t]+VERSION[ \t]+([0-9]+(?:\.[0-9]+)+)[ \t]+LANGUAGES\b')
            if (-not $ProjectVersionMatch.Success) {
                Write-Host "ERROR: Could not determine the package version from CMakeLists.txt." -ForegroundColor Red
                exit 1
            }

            $ProjectVersion = $ProjectVersionMatch.Groups[1].Value
            $PackageBaseName = if ($GpuBackend -eq 'HIP') {
                "LichtFeld-Studio-for-ROCm-$ProjectVersion-windows-x64"
            } else {
                $PackageBackend = $GpuBackend.ToLowerInvariant()
                "LichtFeld-Studio-$ProjectVersion-$PackageBackend-windows-x64"
            }
            $ConfiguredPackageBaseName = $PackageNameMatch.Groups[1].Value
            if ($ConfiguredPackageBaseName -ne $PackageBaseName) {
                Write-Host "ERROR: CPack package name does not match the expected CMake contract." -ForegroundColor Red
                Write-Host "Expected: $PackageBaseName" -ForegroundColor Gray
                Write-Host "Configured: $ConfiguredPackageBaseName" -ForegroundColor Gray
                exit 1
            }

            $PackageArchivePath = Join-Path $BuildDir "$PackageBaseName.zip"
            $PackageChecksumPath = "$PackageArchivePath.sha256"
            $RuntimeManifestPath = if ($GpuBackend -eq 'HIP') {
                Join-Path $BuildDir 'rocm-runtime-manifest.txt'
            } else {
                ''
            }

            Remove-PackageOutputs @($PackageArchivePath, $PackageChecksumPath)
            $PackageGenerationStartedAt = [DateTime]::UtcNow

            Write-Host ""
            Write-Host "Creating portable package..." -ForegroundColor Yellow
            & cmake --build $BuildDir `
                --config $Configuration `
                --target package `
                --parallel 32
            if ($LASTEXITCODE -ne 0) {
                Write-Host "ERROR: Package build failed!" -ForegroundColor Red
                exit 1
            }

            foreach ($ExpectedOutput in @($PackageArchivePath, $PackageChecksumPath)) {
                if (-not (Test-Path -LiteralPath $ExpectedOutput -PathType Leaf)) {
                    Write-Host "ERROR: Package build completed without producing an expected output." -ForegroundColor Red
                    Write-Host "Expected: $ExpectedOutput" -ForegroundColor Gray
                    exit 1
                }
                $OutputInfo = Get-Item -LiteralPath $ExpectedOutput
                if ($OutputInfo.Length -le 0 -or $OutputInfo.LastWriteTimeUtc -lt $PackageGenerationStartedAt) {
                    Write-Host "ERROR: Package output is empty or was not freshly generated." -ForegroundColor Red
                    Write-Host "Output: $ExpectedOutput" -ForegroundColor Gray
                    exit 1
                }
            }
            $PackageArchivePath = (Resolve-Path -LiteralPath $PackageArchivePath).Path
            $PackageChecksumPath = (Resolve-Path -LiteralPath $PackageChecksumPath).Path
            Test-PackageChecksum $PackageArchivePath $PackageChecksumPath
            Test-PackageArchiveContract $PackageArchivePath $GpuBackend $RuntimeManifestPath
        }

        Write-Host ""
        Write-Host "================================================================" -ForegroundColor Green
        Write-Host "Build Successful!" -ForegroundColor Green
        Write-Host "================================================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Executable location:" -ForegroundColor Cyan
        Write-Host "  $ExecutablePath" -ForegroundColor White
        Write-Host ""

        Write-Host "Python module location:" -ForegroundColor Cyan
        Write-Host "  $BuildDir\src\python\$Configuration\lichtfeld.pyd" -ForegroundColor White
        Write-Host ""

        if ($Package) {
            Write-Host "Package archive:" -ForegroundColor Cyan
            Write-Host "  $PackageArchivePath" -ForegroundColor White
            Write-Host "SHA-256 checksum:" -ForegroundColor Cyan
            Write-Host "  $PackageChecksumPath" -ForegroundColor White
            Write-Host ""
        }

    } finally {
        Pop-Location
    }
}

# ============================================================================
# Main Execution
# ============================================================================

$ScriptDisplayName = "LichtFeld-Studio One-Shot Build Script"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host $ScriptDisplayName -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project: $ProjectRoot" -ForegroundColor Gray
Write-Host "Configuration: $Configuration" -ForegroundColor Gray
Write-Host "GPU Backend: $GpuBackend" -ForegroundColor Gray
Write-Host "Python Bindings: Enabled" -ForegroundColor Gray
Write-Host ""

# Phase 1: Environment Verification
if (-not $SkipVerification) {
    $EnvOK = Test-BuildEnvironment
    if (-not $EnvOK) {
        Write-Host "================================================================" -ForegroundColor Red
        Write-Host "Environment Verification Failed" -ForegroundColor Red
        Write-Host "================================================================" -ForegroundColor Red
        Write-Host ""
        Write-Host "Failed checks:" -ForegroundColor Red
        foreach ($Check in $FailedChecks) {
            Write-Host "  - $Check" -ForegroundColor Red
        }
        Write-Host ""
        Write-Host "Please fix the issues above and run this script again." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "Environment Verification Passed!" -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "Skipping environment verification (-SkipVerification)" -ForegroundColor Yellow
    Write-Host ""
}

# Phase 2: Git Submodules
Setup-GitSubmodules

# Phase 3: vcpkg Setup
if (-not $SkipVcpkg) {
    Setup-Vcpkg
} else {
    Write-Host "Skipping vcpkg setup (-SkipVcpkg)" -ForegroundColor Yellow
    # Still set VCPKG_ROOT if vcpkg exists
    if (Test-Path $VcpkgPath) {
        $env:VCPKG_ROOT = $VcpkgPath
    } else {
        Write-Host "WARNING: vcpkg not found at $VcpkgPath" -ForegroundColor Yellow
        Write-Host "Build may fail. Consider running without -SkipVcpkg" -ForegroundColor Yellow
    }
    Write-Host ""
}

# Phase 4: LibTorch Download
if ($GpuBackend -eq 'CUDA' -and -not $SkipLibTorch) {
    Setup-LibTorch
} elseif ($GpuBackend -eq 'HIP') {
    Write-Host "Skipping CUDA LibTorch setup for $GpuBackend backend" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "Skipping LibTorch setup (-SkipLibTorch)" -ForegroundColor Yellow
    Write-Host ""
}

# Phase 5: Build
Build-LichtFeldStudio

Write-Host ""
Write-Host "All done!" -ForegroundColor Green
Write-Host ""
