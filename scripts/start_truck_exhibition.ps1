#Requires -Version 7.0
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
<#
.SYNOPSIS
Open LichtFeld Studio and automatically train the truck dataset in its GUI.
.EXAMPLE
./scripts/start_truck_exhibition.ps1
.EXAMPLE
./scripts/start_truck_exhibition.ps1 -Iterations 2000 -MaxCap 300000 -McpPort 45678
.EXAMPLE
./scripts/start_truck_exhibition.ps1 -Dataset D:/datasets/truck -DryRun
#>
[CmdletBinding()]
param(
    [string]$Dataset = (Join-Path $env:USERPROFILE 'Downloads/tandt_db/tandt/truck'),
    [string]$Executable,
    [string]$OutputRoot,
    [ValidateRange(1, 1000000)][int]$Iterations = 30000,
    [ValidateRange(1000, 10000000)][int]$MaxCap = 1000000,
    [ValidateSet(1, 2, 4, 8)][int]$ResizeFactor = 2,
    [ValidateSet('mrnf', 'mcmc', 'igs+')][string]$Strategy = 'mrnf',
    [ValidateRange(1, 65535)][int]$McpPort = 45678,
    [switch]$Wait,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot
if (-not $IsWindows) { throw 'This launcher requires Windows and PowerShell 7.' }

if ([string]::IsNullOrWhiteSpace($Executable)) {
    $candidates = @(
        $env:LFS_EXECUTABLE,
        (Join-Path $workspace 'build-rocm10/Release/LichtFeld-Studio.exe'),
        (Join-Path $workspace 'build-hip/Release/LichtFeld-Studio.exe'),
        (Join-Path $workspace 'build/Release/LichtFeld-Studio.exe')
    )
    $Executable = $candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } |
        Select-Object -First 1
    if (-not $Executable) { throw 'LichtFeld Studio was not found. Supply -Executable with the built executable path.' }
}
$executablePath = (Resolve-Path -LiteralPath $Executable).ProviderPath
if (-not (Test-Path -LiteralPath $executablePath -PathType Leaf)) { throw "Not an executable file: $executablePath" }
$datasetPath = (Resolve-Path -LiteralPath $Dataset).ProviderPath
if (-not (Test-Path -LiteralPath (Join-Path $datasetPath 'images') -PathType Container)) {
    throw "The dataset must contain an images directory: $datasetPath"
}
$hasColmap = $false
foreach ($subdirectory in @('sparse/0', 'sparse')) {
    foreach ($extension in @('bin', 'txt')) {
        $modelPath = Join-Path $datasetPath $subdirectory
        if ((Test-Path -LiteralPath (Join-Path $modelPath "cameras.$extension") -PathType Leaf) -and
            (Test-Path -LiteralPath (Join-Path $modelPath "images.$extension") -PathType Leaf) -and
            (Test-Path -LiteralPath (Join-Path $modelPath "points3D.$extension") -PathType Leaf)) {
            $hasColmap = $true
        }
    }
}
if (-not $hasColmap) { throw "No complete COLMAP model found in sparse/0 or sparse: $datasetPath" }

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $documents = [Environment]::GetFolderPath([Environment+SpecialFolder]::MyDocuments)
    if ([string]::IsNullOrWhiteSpace($documents)) { $documents = Join-Path $env:USERPROFILE 'Documents' }
    $OutputRoot = Join-Path $documents 'LichtFeld-Exhibition'
}
$outputParent = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputRoot)
$runName = 'truck-{0}-{1}' -f (Get-Date -Format 'yyyyMMdd-HHmmss-fff'), [guid]::NewGuid().ToString('N').Substring(0, 6)
$runOutput = Join-Path $outputParent $runName
$logPath = Join-Path $runOutput 'lichtfeld.log'
$arguments = @(
    '--train', '-d', $datasetPath, '-o', $runOutput,
    '--strategy', $Strategy, '--iter', "$Iterations", '--max-cap', "$MaxCap",
    '--resize_factor', "$ResizeFactor", '--max-width', '0',
    '--mcp-port', "$McpPort", '--log-file', $logPath
)
$plan = [ordered]@{
    executable = $executablePath
    working_directory = Split-Path $executablePath
    dataset = $datasetPath
    output_directory = $runOutput
    log_file = $logPath
    arguments = $arguments
    mcp_endpoint = "http://127.0.0.1:$McpPort/mcp"
}
if ($DryRun) {
    $plan | ConvertTo-Json -Depth 4
    return
}

# Reserve a distinct port for this launch; do not attach to or change an existing GUI.
$listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $McpPort)
$listener.Server.ExclusiveAddressUse = $true
try { $listener.Start() }
catch { throw "MCP port $McpPort is in use. Supply another -McpPort, for example 45679." }
finally { $listener.Stop() }

New-Item -ItemType Directory -Path $runOutput | Out-Null
$plan | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $runOutput 'launch.json') -Encoding utf8
$start = [Diagnostics.ProcessStartInfo]::new()
$start.FileName = $executablePath
$start.WorkingDirectory = Split-Path $executablePath
$start.UseShellExecute = $false
$start.CreateNoWindow = $true # Suppress only the child console; the LFS GUI is visible.
$start.WindowStyle = [Diagnostics.ProcessWindowStyle]::Normal
foreach ($argument in $arguments) { $start.ArgumentList.Add($argument) }

$process = [Diagnostics.Process]::Start($start)
try {
    $plan['process_id'] = $process.Id
    $plan['started_at'] = (Get-Date).ToString('o')
    $plan | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $runOutput 'launch.json') -Encoding utf8
    if ($process.WaitForExit(3000)) {
        throw "LichtFeld Studio exited during startup (code $($process.ExitCode)). See $logPath"
    }
    Write-Host "LichtFeld Studio is opening the truck dataset (PID $($process.Id))."
    Write-Host 'Training starts automatically after import. Progress and the growing model appear in the GUI.'
    Write-Host "Output: $runOutput"
    Write-Host "Log:    $logPath"
    Write-Output ([pscustomobject]@{ ProcessId = $process.Id; OutputDirectory = $runOutput; McpEndpoint = $plan.mcp_endpoint })
    if ($Wait) {
        $process.WaitForExit()
        if ($process.ExitCode -ne 0) { throw "LichtFeld Studio exited with code $($process.ExitCode). See $logPath" }
    }
} finally {
    # Releasing this handle does not close the app; training and viewing continue.
    $process.Dispose()
}
