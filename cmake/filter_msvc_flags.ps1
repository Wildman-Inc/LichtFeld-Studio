#!/usr/bin/env pwsh
# Filter MSVC-style flags that are incompatible with clang when building HIP code
# This script is used as CMAKE_CXX_COMPILER_LAUNCHER to intercept compiler invocations

param(
    [Parameter(Position=0)]
    [string]$Compiler,
    [Parameter(Position=1, ValueFromRemainingArguments=$true)]
    [string[]]$Args
)

# Filter out MSVC-style flags that start with /
$filteredArgs = $Args | Where-Object {
    # Skip MSVC-style flags
    if ($_ -match '^/') {
        Write-Host "Filtering MSVC flag: $_" -ForegroundColor Yellow
        return $false
    }
    return $true
}

# Execute the compiler with filtered arguments
& $Compiler @filteredArgs
exit $LASTEXITCODE
