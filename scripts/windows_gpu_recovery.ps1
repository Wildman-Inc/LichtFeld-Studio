[CmdletBinding()]
param(
    [ValidateSet('Status', 'ApplyStableDefaults', 'EnableHags', 'DisableHags', 'ClearTdr', 'SetExtendedTdr')]
    [string]$Action = 'Status',

    [switch]$Restart
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$graphicsKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers'
$tdrNames = @(
    'TdrDelay',
    'TdrDdiDelay',
    'TdrLevel',
    'TdrDebugMode',
    'TdrLimitTime',
    'TdrLimitCount'
)

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Assert-IsAdmin {
    if (-not (Test-IsAdmin)) {
        throw "Administrative privileges are required for '$Action'. Re-run PowerShell as Administrator."
    }
}

function Get-RegistryValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    try {
        return (Get-ItemProperty -Path $graphicsKey -Name $Name -ErrorAction Stop).$Name
    } catch {
        return $null
    }
}

function Get-DisplayDrivers {
    Get-CimInstance Win32_PnPSignedDriver |
        Where-Object { $_.DeviceClass -eq 'DISPLAY' } |
        Select-Object DeviceName, DriverVersion, DriverDate, Manufacturer, InfName
}

function Show-Status {
    $sessionName = $env:SESSIONNAME
    $hwSchMode = Get-RegistryValue -Name 'HwSchMode'
    $tdrState = [ordered]@{}
    foreach ($name in $tdrNames) {
        $tdrState[$name] = Get-RegistryValue -Name $name
    }

    Write-Host '== Session =='
    [pscustomobject]@{
        User        = $env:USERNAME
        Computer    = $env:COMPUTERNAME
        SessionName = $sessionName
        IsRdp       = ($sessionName -like 'RDP-*')
    } | Format-List | Out-String | Write-Host

    Write-Host '== GraphicsDrivers Registry =='
    [pscustomobject]@{
        HwSchMode = $hwSchMode
    } | Format-List | Out-String | Write-Host

    [pscustomobject]$tdrState | Format-List | Out-String | Write-Host

    Write-Host '== Display Drivers =='
    Get-DisplayDrivers | Format-Table -AutoSize | Out-String | Write-Host
}

function Set-HagsMode {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('Off', 'On')]
        [string]$Mode
    )

    Assert-IsAdmin

    $value = if ($Mode -eq 'Off') { 1 } else { 2 }
    New-ItemProperty -Path $graphicsKey -Name 'HwSchMode' -PropertyType DWord -Value $value -Force | Out-Null
    Write-Host "Set HwSchMode=$value ($Mode). Reboot required."
}

function Clear-TdrOverrides {
    Assert-IsAdmin

    foreach ($name in $tdrNames) {
        Remove-ItemProperty -Path $graphicsKey -Name $name -ErrorAction SilentlyContinue
    }

    Write-Host 'Removed TDR override values. Reboot required.'
}

function Set-ExtendedTdr {
    Assert-IsAdmin

    New-ItemProperty -Path $graphicsKey -Name 'TdrDelay' -PropertyType DWord -Value 10 -Force | Out-Null
    New-ItemProperty -Path $graphicsKey -Name 'TdrDdiDelay' -PropertyType DWord -Value 25 -Force | Out-Null
    Write-Host 'Set TdrDelay=10 and TdrDdiDelay=25. Reboot required.'
}

switch ($Action) {
    'Status' {
        Show-Status
    }
    'ApplyStableDefaults' {
        Set-HagsMode -Mode Off
        Clear-TdrOverrides
        Show-Status
    }
    'EnableHags' {
        Set-HagsMode -Mode On
        Show-Status
    }
    'DisableHags' {
        Set-HagsMode -Mode Off
        Show-Status
    }
    'ClearTdr' {
        Clear-TdrOverrides
        Show-Status
    }
    'SetExtendedTdr' {
        Set-ExtendedTdr
        Show-Status
    }
    default {
        throw "Unknown action: $Action"
    }
}

if ($Restart.IsPresent -and $Action -ne 'Status') {
    Assert-IsAdmin
    Restart-Computer -Force
}
