param(
    [Parameter(Mandatory=$true)][string]$BaseExecutable,
    [Parameter(Mandatory=$true)][string]$CombinedExecutable,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [string]$BaseSha = '69d080dc9995003cef7d3fea68d7789f3eb2a0ad',
    [string]$CombinedSha = 'cf312e294ccb6208af502bdb7df018f0607c5a02',
    [int]$Rounds = 6,
    [int]$Warmups = 50,
    [int]$Samples = 50,
    [switch]$SnapshotOnly,
    [Parameter(Mandatory=$true)][string]$PythonSite,
    [Parameter(Mandatory=$true)][string]$PrivatePrefix,
    [Parameter(Mandatory=$true)][string]$VcpkgBin
)
$ErrorActionPreference = 'Stop'
$baseExe = (Resolve-Path -LiteralPath $BaseExecutable).Path
$combinedExe = (Resolve-Path -LiteralPath $CombinedExecutable).Path
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$sdkBin = Join-Path $pythonSite '_rocm_sdk_core\bin'
$privateBin = Join-Path $PrivatePrefix 'bin'
function Invoke-Variant([string]$Executable, [string]$Label, [string]$Sha, [string]$Scene, [int]$Round) {
    $exe = (Resolve-Path -LiteralPath $Executable).Path
    $bin = Split-Path -Parent $exe
    $buildRoot = Split-Path -Parent (Split-Path -Parent $bin)
    $buildRelease = Join-Path $buildRoot 'Release'
    $dll = Join-Path $bin 'amdhip64_7.dll'
    $held = "$dll.integrated-held"
    $priorPath = $env:PATH
    $priorSnapshot = $env:LFS_BENCH_SNAPSHOT_ONLY
    $heldDll = $false
    try {
        if (Test-Path -LiteralPath $dll) {
            if (Test-Path -LiteralPath $held) { throw "Refusing to overwrite held DLL $held" }
            Move-Item -LiteralPath $dll -Destination $held
            $heldDll = $true
        }
        if (-not (Test-Path -LiteralPath $buildRelease)) { throw "Expected variant build Release directory missing: $buildRelease" }
        $env:PATH = ((@($bin, $buildRelease, $sdkBin, $privateBin, (Join-Path $pythonSite 'torch\lib'), (Join-Path $pythonSite '_rocm_sdk_devel\bin'), (Join-Path $pythonSite '_rocm_sdk_libraries\bin'), $vcpkgBin) -join ';') + ";$priorPath")
        if ($SnapshotOnly) { $env:LFS_BENCH_SNAPSHOT_ONLY = '1' } else { Remove-Item Env:LFS_BENCH_SNAPSHOT_ONLY -ErrorAction SilentlyContinue }
        $snapshot = Join-Path $OutputDirectory ("{0}-{1}-round{2}" -f $Scene,$Label,$Round)
        $log = "$snapshot.log"
        $exeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash
        Set-Content -LiteralPath $log -Value ("EXEC variant={0} sha256={1} exe={2} scene={3} round={4} warmups={5} samples={6}" -f $Label,$exeHash,$exe,$Scene,$Round,$Warmups,$Samples)
        $cliArgs = @('safe', $Warmups, $Samples, $Scene, $snapshot, $Sha)
        if ($Label -eq 'combined-native') { $cliArgs[0] = 'native' }
        & $exe @cliArgs 2>&1 | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) { throw "$Label $Scene round $Round failed with exit code $LASTEXITCODE" }
    }
    finally {
        $env:PATH = $priorPath
        if ($null -eq $priorSnapshot) { Remove-Item Env:LFS_BENCH_SNAPSHOT_ONLY -ErrorAction SilentlyContinue } else { $env:LFS_BENCH_SNAPSHOT_ONLY = $priorSnapshot }
        if ($heldDll -and (Test-Path -LiteralPath $held)) { Move-Item -LiteralPath $held -Destination $dll }
    }
}
if ($SnapshotOnly) {
    foreach ($scene in @('half','full')) {
        Invoke-Variant $baseExe 'base-safe' $BaseSha $scene 0
        Invoke-Variant $combinedExe 'combined-safe' $CombinedSha $scene 0
        Invoke-Variant $combinedExe 'combined-native' $CombinedSha $scene 0
    }
    exit 0
}
$variants = @(
    @{ Label='base-safe'; Exe=$baseExe; Sha=$BaseSha },
    @{ Label='combined-safe'; Exe=$combinedExe; Sha=$CombinedSha },
    @{ Label='combined-native'; Exe=$combinedExe; Sha=$CombinedSha }
)
foreach ($scene in @('half','full')) {
    for ($round = 1; $round -le $Rounds; $round++) {
        $rotation = ($round - 1) % $variants.Count
        $order = @($variants[$rotation..($variants.Count - 1)] + $variants[0..($rotation - 1)])
        if ($rotation -eq 0) { $order = $variants }
        foreach ($variant in $order) {
            Invoke-Variant $variant.Exe $variant.Label $variant.Sha $scene $round
        }
    }
}
