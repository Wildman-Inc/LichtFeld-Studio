# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
# Compare two self-contained Release runtimes without replacing either one.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Dataset,
    [Parameter(Mandatory)][string]$BaselineRuntime,
    [Parameter(Mandatory)][string]$CandidateRuntime,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [string]$BaselineCommit,
    [ValidateRange(1200, 9999)][int]$Iterations = 2000,
    [ValidateRange(100000, 1000000)][int]$MaxCap = 100000,
    [ValidateRange(1, 100)][int]$Repetitions = 4,
    [ValidateSet(1, 2)][int[]]$ResizeFactors = @(1, 2),
    [switch]$DisableRocJpeg,
    [switch]$DisableAsyncAllocator,
    [switch]$CollectProfile,
    [switch]$Densify,
    [switch]$Evaluate
)
$ErrorActionPreference = 'Stop'
if (-not $Densify -and $MaxCap -ne 100000) { throw 'The fixed workload requires MaxCap=100000' }
$workspace = Split-Path $PSScriptRoot
$datasetPath = (Resolve-Path -LiteralPath $Dataset).Path
$runtimes = @{
    baseline=(Resolve-Path -LiteralPath $BaselineRuntime).Path
    candidate=(Resolve-Path -LiteralPath $CandidateRuntime).Path
}
foreach ($runtime in $runtimes.Values) {
    if (-not (Test-Path -LiteralPath (Join-Path $runtime 'LichtFeld-Studio.exe'))) { throw "Missing runtime: $runtime" }
}
foreach ($name in @('rocjpeg.dll', 'amdhip64_7.dll')) {
    if ((Get-FileHash -LiteralPath (Join-Path $runtimes.baseline $name)).Hash -ne
        (Get-FileHash -LiteralPath (Join-Path $runtimes.candidate $name)).Hash) {
        throw "Comparison requires the same $name"
    }
}
if (Test-Path -LiteralPath $OutputDirectory) { throw "Results already exist: $OutputDirectory" }
$results = (New-Item -ItemType Directory -Path $OutputDirectory).FullName
$work = Join-Path $results 'training-work'
New-Item -ItemType Directory -Path $work | Out-Null
$configPath = Join-Path $work 'config.json'
$configuration = @{ optimization = @{
    strategy='mrnf'; iterations=$Iterations; sh_degree=3; sh_degree_interval=1;
    means_lr=0.00002; shs_lr=0.002; opacity_lr=0.012; scaling_lr=0.007;
    rotation_lr=0.002; lambda_dssim=0.2; refine_every=100;
    start_refine=$(if ($Densify) {500} else {10000}); stop_refine=20000;
    min_opacity=(1.0/255.0); save_steps=@(); eval_steps=@(); morton_reorder_interval=0
} }
$configuration | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $configPath -Encoding utf8
$records = [Collections.Generic.List[object]]::new()
$metadata = [ordered]@{
    timestamp=(Get-Date).ToString('o'); dataset=$datasetPath;
    baseline_commit=$BaselineCommit; candidate_commit=(& git -C $workspace rev-parse HEAD).Trim();
    candidate_changes=@(& git -C $workspace diff --name-only);
    config=$configuration; iterations=$Iterations; max_cap=$MaxCap; repetitions=$Repetitions;
    rocjpeg_enabled=(-not $DisableRocJpeg); async_allocator_disabled=[bool]$DisableAsyncAllocator;
    collect_profile=[bool]$CollectProfile; densify=[bool]$Densify; evaluate=[bool]$Evaluate;
    warmup_excluded_iterations=$(if ($CollectProfile) {1100} else {0}); external_warmup_iterations=300;
    cache='default application cache; warm OS file cache'; resize_filter='Lanczos2';
    order='sequential GPU execution; alternating baseline/candidate order';
    processor=@(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors);
    video=@(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion);
    runtimes=@{}; runs=$records
}
foreach ($variant in @('baseline','candidate')) {
    $metadata.runtimes[$variant] = [ordered]@{
        directory=$runtimes[$variant];
        files=@(Get-ChildItem -LiteralPath $runtimes[$variant] -File |
            Where-Object { $_.Extension -in '.exe','.dll','.pyd' } |
            ForEach-Object { @{ name=$_.Name; sha256=(Get-FileHash -LiteralPath $_.FullName).Hash } })
    }
}

function Remove-TrainingWork([string]$Path) {
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if ($resolved -ne $work -and -not $resolved.StartsWith($work + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Cleanup escaped the training work directory'
    }
    if ((Get-Item -LiteralPath $resolved).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Work is a reparse point' }
    if (@(Get-ChildItem -LiteralPath $resolved -Force -Recurse -Attributes ReparsePoint).Count) { throw 'Work contains a reparse point' }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}

function Invoke-TrainingRun([string]$Variant, [int]$Resize, [int]$Repetition, [bool]$Warmup) {
    $kind = if ($Warmup) {'warmup'} else {"r$Repetition"}
    $name = "resize${Resize}_${Variant}_${kind}"
    $runOutput = Join-Path $work $name
    $count = if ($Warmup) {300} else {$Iterations}
    $arguments = @('--headless','-d',$datasetPath,'-o',$runOutput,'--config',$configPath,
        '--iter',"$count",'--max-cap',"$MaxCap",'--resize_factor',"$Resize",'--max-width','0',
        '--sh-degree-interval','1')
    if ($CollectProfile) { $arguments += @('--perf-bench','--perf-bench-warmup',$(if ($Warmup) {'100'} else {'1100'})) }
    if ($Evaluate -and -not $Warmup) { $arguments += @('--eval','--test-every','8','--eval-steps',"$count",'--no-save-eval-images') }
    $start = [Diagnostics.ProcessStartInfo]::new()
    $start.FileName = Join-Path $runtimes[$Variant] 'LichtFeld-Studio.exe'
    $start.WorkingDirectory = $runtimes[$Variant]
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.WindowStyle = [Diagnostics.ProcessWindowStyle]::Hidden
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.Environment['LFS_DISABLE_ROCJPEG'] = if ($DisableRocJpeg) {'1'} else {'0'}
    $start.Environment['LFS_DISABLE_ASYNC_ALLOCATOR'] = if ($DisableAsyncAllocator) {'1'} else {'0'}
    foreach ($argument in $arguments) { $start.ArgumentList.Add($argument) }
    Write-Output "START $name ($count iterations)"
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $process = [Diagnostics.Process]::Start($start)
    try {
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(600000)) {
            $process.Kill(); $process.WaitForExit()
            throw "Timed out: $name"
        }
        $timer.Stop()
        $log = $stdout.GetAwaiter().GetResult() + $stderr.GetAwaiter().GetResult()
        $log | Set-Content -LiteralPath (Join-Path $results "$name.log") -Encoding utf8
        if ($process.ExitCode -ne 0) { throw "Failed: $name exit $($process.ExitCode)" }
        $training = [regex]::Match($log, 'Training completed in ([0-9.]+)s')
        $decodes = [regex]::Match($log, 'rocJPEG hardware decodes: (\d+), CPU decodes: (\d+)')
        $splats = [regex]::Match($log, 'Final splats: (\d+)')
        $loaded = [regex]::Match($log, 'Done: (\d+) loaded, (\d+) hits, (\d+) misses')
        if (-not $training.Success -or -not $decodes.Success -or -not $splats.Success -or -not $loaded.Success) { throw "Missing counters: $name" }
        $hw = [int]$decodes.Groups[1].Value
        $cpu = [int]$decodes.Groups[2].Value
        if (($DisableRocJpeg -and ($hw -ne 0 -or $cpu -lt $count)) -or
            (-not $DisableRocJpeg -and ($hw -lt $count -or $cpu -ne 0))) { throw "Unexpected decoder: $name" }
        if (-not $Densify -and [int]$splats.Groups[1].Value -ne 100000) { throw "Unexpected splat count: $name" }
        if ($log -match '\[error\]|Loss: (nan|inf)') { throw "Runtime error: $name" }
        $perf = if ($CollectProfile) { Get-Content -LiteralPath (Join-Path $runOutput 'perf_bench.json') -Raw | ConvertFrom-Json } else { $null }
        $metrics = @(Get-ChildItem -LiteralPath $runOutput -Recurse -File -Filter '*metrics.csv' |
            ForEach-Object { @{ name=$_.FullName.Substring($runOutput.Length + 1); rows=@(Import-Csv -LiteralPath $_.FullName) } })
        if ($Evaluate -and -not $Warmup -and $metrics.Count -eq 0) { throw "Missing evaluation metrics: $name" }
        $records.Add([ordered]@{
            name=$name; variant=$Variant; resize_factor=$Resize; repetition=$Repetition; warmup=$Warmup;
            command_arguments=$arguments; elapsed_seconds=$timer.Elapsed.TotalSeconds;
            process_cpu_seconds=$process.TotalProcessorTime.TotalSeconds;
            training_seconds=[double]::Parse($training.Groups[1].Value,[Globalization.CultureInfo]::InvariantCulture);
            hardware_decodes=$hw; cpu_decodes=$cpu; final_splats=[int]$splats.Groups[1].Value;
            loaded_images=[int]$loaded.Groups[1].Value; hot_cache_hits=[int]$loaded.Groups[2].Value;
            losses=@([regex]::Matches($log, 'Loss: ([0-9.]+)') | ForEach-Object { [double]::Parse($_.Groups[1].Value,[Globalization.CultureInfo]::InvariantCulture) });
            perf=$perf; evaluation=$metrics; log_file="$name.log"
        })
        $metadata | ConvertTo-Json -Depth 45 | Set-Content -LiteralPath (Join-Path $results 'results.json') -Encoding utf8
        Write-Output ("DONE {0}: train={1}s wall={2:F3}s" -f $name,$training.Groups[1].Value,$timer.Elapsed.TotalSeconds)
    } finally {
        $process.Dispose()
    }
    Remove-TrainingWork $runOutput
}

try {
    foreach ($resize in $ResizeFactors) {
        foreach ($variant in @('baseline','candidate')) { Invoke-TrainingRun $variant $resize 0 $true }
        for ($repetition=1; $repetition -le $Repetitions; ++$repetition) {
            $order = if ($repetition % 2) { @('baseline','candidate') } else { @('candidate','baseline') }
            foreach ($variant in $order) { Invoke-TrainingRun $variant $resize $repetition $false }
        }
    }
} finally {
    Remove-TrainingWork $work
}
Write-Output 'Comparison finished; temporary configuration and training projects removed.'
