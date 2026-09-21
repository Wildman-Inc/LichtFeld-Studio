# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
# Run after building the native Windows rocJPEG backend and CPU Lanczos2 path.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Dataset,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [string]$BuildDirectory = (Join-Path (Split-Path $PSScriptRoot) 'build-rocm10'),
    [ValidateRange(1200, 9999)][int]$Iterations = 2000,
    [ValidateRange(1, 100)][int]$Repetitions = 4,
    [switch]$CollectProfile
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot
$build = (Resolve-Path -LiteralPath $BuildDirectory).Path
$datasetPath = (Resolve-Path -LiteralPath $Dataset).Path
$executable = Join-Path $build 'Release\LichtFeld-Studio.exe'
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) { throw "Missing executable: $executable" }
if (Test-Path -LiteralPath $OutputDirectory) { throw "Result directory already exists: $OutputDirectory" }
$work = Join-Path $build 'rocjpeg-benchmark-work'
if (Test-Path -LiteralPath $work) { throw "Benchmark work directory already exists: $work" }
$results = (New-Item -ItemType Directory -Path $OutputDirectory).FullName
New-Item -ItemType Directory -Path $work | Out-Null
$configPath = Join-Path $work 'config.json'
$configuration = @{ optimization = @{
    strategy='mrnf'; iterations=$Iterations; sh_degree=3; sh_degree_interval=1;
    means_lr=0.00002; shs_lr=0.002; opacity_lr=0.012; scaling_lr=0.007;
    rotation_lr=0.002; lambda_dssim=0.2; refine_every=100;
    start_refine=10000; stop_refine=20000; min_opacity=(1.0/255.0);
    save_steps=@(); eval_steps=@(); morton_reorder_interval=0
} }
$configuration | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $configPath -Encoding utf8
$records = [Collections.Generic.List[object]]::new()
$metadata = [ordered]@{
    timestamp=(Get-Date).ToString('o'); dataset=$datasetPath;
    executable=$executable; executable_sha256=(Get-FileHash -LiteralPath $executable).Hash;
    core_sha256=(Get-FileHash -LiteralPath (Join-Path (Split-Path $executable) 'lfs_core.dll')).Hash;
    rocjpeg_sha256=(Get-FileHash -LiteralPath (Join-Path (Split-Path $executable) 'rocjpeg.dll')).Hash;
    hip_runtime_sha256=(Get-FileHash -LiteralPath (Join-Path (Split-Path $executable) 'amdhip64_7.dll')).Hash;
    source_commit=(& git -C $workspace rev-parse HEAD).Trim();
    source_changes=@(& git -C $workspace diff --name-only);
    resize_source_sha256=(Get-FileHash -LiteralPath (Join-Path $workspace 'src\core\image_io.cpp')).Hash;
    rocjpeg_commit=(& git -C (Join-Path $build '_deps\rocjpeg-src') rev-parse HEAD).Trim();
    processor=@(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors);
    video=@(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion);
    config=$configuration; repetitions=$Repetitions; iterations=$Iterations;
    collect_profile=[bool]$CollectProfile;
    warmup_excluded_iterations=$(if ($CollectProfile) {1100} else {0}); external_warmup_iterations=300;
    resize_filter='Lanczos2 on CPU and GPU; no resizing for native resolution';
    cache='default application cache; warm OS file cache';
    order='sequential GPU execution; alternate ON/OFF pair order by repetition';
    runs=$records
}

function Remove-BenchmarkOutput([string]$Path) {
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if (-not $resolved.StartsWith($work + '\', [StringComparison]::OrdinalIgnoreCase) -and $resolved -ne $work) {
        throw 'Cleanup path escaped the benchmark work directory'
    }
    if ((Get-Item -LiteralPath $resolved).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Cleanup target is a reparse point' }
    if (@(Get-ChildItem -LiteralPath $resolved -Force -Recurse -Attributes ReparsePoint).Count) { throw 'Cleanup contains a reparse point' }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}

function Invoke-BenchmarkRun([string]$Case, [int]$Resize, [bool]$Enabled, [int]$Repetition, [bool]$Warmup) {
    $mode = if ($Enabled) {'on'} else {'off'}
    $kind = if ($Warmup) {'warmup'} else {"r$Repetition"}
    $name = "${Case}_${mode}_${kind}"
    $runOutput = Join-Path $work $name
    $count = if ($Warmup) {300} else {$Iterations}
    $excluded = if ($Warmup) {100} else {1100}
    Write-Output "START $name ($count iterations)"
    $arguments = @('--headless','-d',$datasetPath,'-o',$runOutput,'--config',$configPath,
        '--iter',"$count",'--max-cap','100000','--resize_factor',"$Resize",'--max-width','0',
        '--sh-degree-interval','1')
    if ($CollectProfile) { $arguments += @('--perf-bench','--perf-bench-warmup',"$excluded") }
    $start = [Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $executable
    $start.WorkingDirectory = Split-Path $executable
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.WindowStyle = [Diagnostics.ProcessWindowStyle]::Hidden
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.Environment['LFS_DISABLE_ROCJPEG'] = if ($Enabled) {'0'} else {'1'}
    foreach ($argument in $arguments) { $start.ArgumentList.Add($argument) }
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $process = [Diagnostics.Process]::Start($start)
    try {
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(600000)) {
            $process.Kill()
            $process.WaitForExit()
            throw "Benchmark timed out: $name"
        }
        $timer.Stop()
        $log = $stdout.GetAwaiter().GetResult() + $stderr.GetAwaiter().GetResult()
        $log | Set-Content -LiteralPath (Join-Path $results "$name.log") -Encoding utf8
        if ($process.ExitCode -ne 0) { throw "Benchmark failed: $name exit $($process.ExitCode)" }
        $perf = if ($CollectProfile) { Get-Content -LiteralPath (Join-Path $runOutput 'perf_bench.json') -Raw | ConvertFrom-Json } else { $null }
        $training = [regex]::Match($log, 'Training completed in ([0-9.]+)s')
        $decodes = [regex]::Match($log, 'rocJPEG hardware decodes: (\d+), CPU decodes: (\d+)')
        $splats = [regex]::Match($log, 'Final splats: (\d+)')
        $loaded = [regex]::Match($log, 'Done: (\d+) loaded, (\d+) hits, (\d+) misses')
        if (-not $training.Success -or -not $decodes.Success -or -not $splats.Success -or -not $loaded.Success) { throw "Missing run statistics: $name" }
        $hardware = [int]$decodes.Groups[1].Value
        $cpu = [int]$decodes.Groups[2].Value
        $available = $log.Contains('Windows D3D11/VCN hardware JPEG enabled')
        if (($Enabled -and (-not $available -or $hardware -lt $count -or $cpu -ne 0)) -or
            (-not $Enabled -and ($available -or $hardware -ne 0 -or $cpu -lt $count)) -or
            [int]$loaded.Groups[1].Value -lt $count) { throw "Unexpected decoder selection: $name HW=$hardware CPU=$cpu" }
        if ([int]$splats.Groups[1].Value -ne 100000 -or ($CollectProfile -and [int]$perf.total_iters -ne $count)) { throw "Unexpected training workload: $name" }
        $records.Add([ordered]@{
            name=$name; case=$Case; enabled=$Enabled; repetition=$Repetition; warmup=$Warmup;
            command_arguments=$arguments; elapsed_seconds=$timer.Elapsed.TotalSeconds;
            process_cpu_seconds=$process.TotalProcessorTime.TotalSeconds;
            training_seconds=[double]::Parse($training.Groups[1].Value,[Globalization.CultureInfo]::InvariantCulture);
            hardware_decodes=$hardware; cpu_decodes=$cpu; loaded_images=[int]$loaded.Groups[1].Value;
            hot_cache_hits=[int]$loaded.Groups[2].Value; final_splats=[int]$splats.Groups[1].Value;
            perf=$perf; log_file="$name.log"
        })
        $metadata | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath (Join-Path $results 'results.json') -Encoding utf8
        Write-Output ("DONE {0}: train={1:F3}s wall={2:F3}s HW={3} CPU={4}" -f
            $name,[double]$training.Groups[1].Value,$timer.Elapsed.TotalSeconds,$hardware,$cpu)
    } finally {
        $process.Dispose()
    }
    Remove-BenchmarkOutput $runOutput
}

try {
    foreach ($case in @(@{Name='native'; Resize=1},@{Name='half'; Resize=2})) {
        Invoke-BenchmarkRun $case.Name $case.Resize $true 0 $true
        Invoke-BenchmarkRun $case.Name $case.Resize $false 0 $true
        for ($repetition=1; $repetition -le $Repetitions; ++$repetition) {
            $order = if (($repetition % 2) -eq 1) {@($true,$false)} else {@($false,$true)}
            foreach ($enabled in $order) { Invoke-BenchmarkRun $case.Name $case.Resize $enabled $repetition $false }
        }
    }
} finally {
    Remove-BenchmarkOutput $work
}
Write-Output 'Benchmark finished; temporary configuration and training projects removed.'
