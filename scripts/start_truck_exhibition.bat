@echo off
setlocal
set "LFS_PWSH=%ProgramFiles%\PowerShell\7\pwsh.exe"
if exist "%LFS_PWSH%" goto launch
set "LFS_PWSH=pwsh.exe"
where pwsh.exe >nul 2>nul
if errorlevel 1 (
    echo PowerShell 7 is required. Install it or launch start_truck_exhibition.ps1 from PowerShell 7.
    pause
    exit /b 1
)
:launch
"%LFS_PWSH%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_truck_exhibition.ps1" %*
if errorlevel 1 (
    echo Exhibition launch failed. See the error above.
    pause
    exit /b 1
)
exit /b 0
