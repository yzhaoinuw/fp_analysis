@echo off
setlocal

set "APP_DIR=%~dp0"
set "FP_ANALYSIS_APP_DIR=%APP_DIR%"
set "NO_PAUSE="
if /I "%~1"=="--smoke" set "NO_PAUSE=1"

if not exist "%APP_DIR%run_fp_analysis_app.exe" (
    echo Cannot find run_fp_analysis_app.exe next to this launcher.
    echo Move this launcher back into the unzipped app folder.
    echo.
    pause
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $AppRoot = $env:FP_ANALYSIS_APP_DIR; function Invoke-UnblockTarget { param([string]$Path) if (-not (Test-Path -LiteralPath $Path)) { return }; $Item = Get-Item -LiteralPath $Path -Force; if ($Item.PSIsContainer) { Get-ChildItem -LiteralPath $Path -Recurse -Force -File | ForEach-Object { Unblock-File -LiteralPath $_.FullName -ErrorAction Stop } } else { Unblock-File -LiteralPath $Item.FullName -ErrorAction Stop } }; if (Get-Command Unblock-File -ErrorAction SilentlyContinue) { Write-Host 'Preparing FP Analysis App files...'; try { 'run_fp_analysis_app.exe','unblock_app.cmd','_internal','fp_analysis_app' | ForEach-Object { Invoke-UnblockTarget -Path (Join-Path $AppRoot $_) } } catch { Write-Warning ('Some files could not be unblocked automatically: ' + $_.Exception.Message); Write-Warning 'The app will still try to start.' } }"

echo Starting FP Analysis App...
pushd "%APP_DIR%"
"%APP_DIR%run_fp_analysis_app.exe" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" (
    echo.
    echo FP Analysis App did not start successfully.
    echo Leave this window open and send Yue the message above.
    if not defined NO_PAUSE pause
)

exit /b %EXIT_CODE%
