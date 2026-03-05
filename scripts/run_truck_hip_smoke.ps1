[CmdletBinding()]
param(
    [string]$BuildDir = 'C:\Dev\LichtFeld-Studio\build-hip',
    [string]$DataPath = 'C:\Users\HarutoWatanabe\Downloads\tandt_db\tandt\truck',
    [string]$OutputRoot = 'C:\Users\HarutoWatanabe\Downloads',
    [int]$Iterations = 300,
    [switch]$AllowRdp,
    [switch]$PrintOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$sessionName = $env:SESSIONNAME
if (-not $AllowRdp.IsPresent -and $sessionName -like 'RDP-*') {
    throw "Refusing to run under RDP session '$sessionName'. Log in on the local console or pass -AllowRdp to override."
}

$exe = Join-Path $BuildDir 'LichtFeld-Studio.exe'
if (-not (Test-Path $exe)) {
    throw "Executable not found: $exe"
}

if (-not (Test-Path $DataPath)) {
    throw "Dataset path not found: $DataPath"
}

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$outputDir = Join-Path $OutputRoot "lfs_truck_hip_smoke_$timestamp"
$trainLog = Join-Path $outputDir 'train.log'
$stdoutLog = Join-Path $outputDir 'stdout.log'
$stderrLog = Join-Path $outputDir 'stderr.log'

$args = @(
    '--headless'
    '--train'
    '--no-splash'
    "--data-path=$DataPath"
    "--output-path=$outputDir"
    "--iter=$Iterations"
    '--strategy=mcmc'
    '--tile-mode=4'
    '--max-cap=140000'
    '--resize_factor=8'
    '--log-level=info'
    "--log-file=$trainLog"
)

$commandLine = @($exe) + $args
Write-Host '== Output Directory =='
Write-Host $outputDir
Write-Host '== Command =='
Write-Host ($commandLine -join ' ')

if ($PrintOnly.IsPresent) {
    return
}

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$process = Start-Process -FilePath $exe `
    -ArgumentList $args `
    -WorkingDirectory $BuildDir `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Write-Host "Started PID $($process.Id)"
Write-Host "stdout: $stdoutLog"
Write-Host "stderr: $stderrLog"
Write-Host "train:  $trainLog"
