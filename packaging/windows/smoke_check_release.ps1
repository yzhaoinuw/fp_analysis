param(
    [Parameter(Mandatory = $true)]
    [string]$Path
)

$ErrorActionPreference = "Stop"

$ReleasePath = (Resolve-Path -LiteralPath $Path).Path

function Assert-Exists {
    param([string]$RelativePath)

    $FullPath = Join-Path $ReleasePath $RelativePath
    if (-not (Test-Path -LiteralPath $FullPath)) {
        throw "Missing expected release item: $RelativePath"
    }
}

function Assert-Any {
    param(
        [string]$RelativePath,
        [string]$Filter
    )

    $FullPath = Join-Path $ReleasePath $RelativePath
    if (-not (Test-Path -LiteralPath $FullPath)) {
        throw "Missing expected release directory: $RelativePath"
    }

    $Matches = Get-ChildItem -LiteralPath $FullPath -Filter $Filter -File
    if (-not $Matches) {
        throw "No files matching $Filter found under $RelativePath"
    }
}

Assert-Exists "_internal"
Assert-Exists "run_fp_analysis_app.exe"
Assert-Exists "unblock_app.cmd"
Assert-Exists "fp_analysis_app"
Assert-Exists "fp_analysis_app\__init__.py"
Assert-Exists "fp_analysis_app\app_dev.py"
Assert-Exists "fp_analysis_app\pages"
Assert-Exists "fp_analysis_app\assets\figures"
Assert-Exists "fp_analysis_app\assets\spreadsheets"
Assert-Exists "fp_analysis_app\assets\videos"
Assert-Any "fp_analysis_app\pages" "*.py"

Write-Host "Release structure smoke check passed: $ReleasePath"
