param(
    [string]$BuildPython = "",
    [string]$TestPython = "",
    [switch]$SkipTests,
    [switch]$AllowDirty
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..\..")).Path
$ArtifactDir = Join-Path $Repo "release_artifacts"

Set-Location $Repo
$env:FP_ANALYSIS_REPO_ROOT = $Repo

function Resolve-Python {
    param(
        [string]$ExplicitPath,
        [string]$EnvironmentName
    )

    if ($ExplicitPath) {
        return (Resolve-Path -LiteralPath $ExplicitPath).Path
    }

    $Candidate = Join-Path $env:USERPROFILE "miniconda3\envs\$EnvironmentName\python.exe"
    if (-not (Test-Path -LiteralPath $Candidate)) {
        throw "Could not find $EnvironmentName Python at $Candidate. Pass an explicit Python path."
    }
    return (Resolve-Path -LiteralPath $Candidate).Path
}

function Invoke-Native {
    param(
        [string]$FilePath,
        [string[]]$CommandArgs
    )

    & $FilePath @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($CommandArgs -join ' ')"
    }
}

function Invoke-NativeCapture {
    param(
        [string]$FilePath,
        [string[]]$CommandArgs
    )

    $Output = & $FilePath @CommandArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($CommandArgs -join ' ')`n$($Output | Out-String)"
    }
    return ($Output | Out-String).Trim()
}

function Assert-RepoChildPath {
    param([string]$Path)

    $RepoPrefix = $Repo.TrimEnd("\") + "\"
    $FullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $FullPath.StartsWith($RepoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify a path outside the repository: $FullPath"
    }
    return $FullPath
}

function Remove-RepoBuildPath {
    param([string]$Path)

    $SafePath = Assert-RepoChildPath -Path $Path
    if (Test-Path -LiteralPath $SafePath) {
        Remove-Item -LiteralPath $SafePath -Recurse -Force
    }
}

$BuildPython = Resolve-Python -ExplicitPath $BuildPython -EnvironmentName "fp_analysis_dist"
$TestPython = Resolve-Python -ExplicitPath $TestPython -EnvironmentName "fiber_photometry"

$TrackedStatus = Invoke-NativeCapture -FilePath "git" -CommandArgs @(
    "status",
    "--short",
    "--untracked-files=no"
)
if (-not $AllowDirty -and $TrackedStatus) {
    throw "Tracked files are modified. Commit, stash, or rerun with -AllowDirty for a local test build."
}

$Version = Invoke-NativeCapture -FilePath $BuildPython -CommandArgs @(
    "-c",
    "from fp_analysis_app import VERSION; print(VERSION)"
)
$Version = $Version.Trim()

$PackageFolderName = Invoke-NativeCapture -FilePath $BuildPython -CommandArgs @(
    "packaging\windows\package_folder_name.py",
    $Version
)
$ArtifactBaseName = "fp_analysis_app_$Version"
$DistPath = Join-Path $Repo "dist\$PackageFolderName"
$ZipPath = Join-Path $ArtifactDir "$ArtifactBaseName-windows.zip"
$SpecPath = Join-Path $ScriptDir "app.spec"

New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null

Write-Host "Building $ArtifactBaseName into folder $PackageFolderName"
Write-Host "Build Python: $BuildPython"
Write-Host "Test Python:  $TestPython"

Invoke-Native -FilePath $BuildPython -CommandArgs @("-m", "pip", "check")
Invoke-Native -FilePath $BuildPython -CommandArgs @(
    "-c",
    "import desktop_app_source_updater, webview; print('packaging imports ok')"
)

if (-not $SkipTests) {
    Invoke-Native -FilePath $TestPython -CommandArgs @(
        "-m",
        "unittest",
        "tests.test_perievent_analysis",
        "tests.test_startup_update",
        "tests.test_packaging"
    )
}

Remove-RepoBuildPath -Path $DistPath

Invoke-Native -FilePath $BuildPython -CommandArgs @(
    "-m",
    "PyInstaller",
    "--clean",
    "--noconfirm",
    $SpecPath
)

if (-not (Test-Path -LiteralPath $DistPath)) {
    throw "PyInstaller did not create the expected folder: $DistPath"
}

$RuntimePath = Join-Path $DistPath "fp_analysis_app"
Remove-RepoBuildPath -Path $RuntimePath

$ExportArgs = @(
    "packaging\windows\export_runtime_from_git.py",
    "--repo",
    $Repo,
    "--runtime-path",
    "fp_analysis_app",
    "--destination",
    $DistPath
)
if ($AllowDirty) {
    $ExportArgs += "--worktree"
} else {
    $ExportArgs += @("--ref", "HEAD")
}
Invoke-Native -FilePath $BuildPython -CommandArgs $ExportArgs

foreach ($RelativePath in @(
    "fp_analysis_app\assets\figures",
    "fp_analysis_app\assets\spreadsheets",
    "fp_analysis_app\assets\videos"
)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $DistPath $RelativePath) |
        Out-Null
}

Copy-Item -LiteralPath (Join-Path $ScriptDir "release_helpers\unblock_app.cmd") `
    -Destination $DistPath -Force

Get-ChildItem -LiteralPath $DistPath -Directory -Recurse -Filter "__pycache__" |
    ForEach-Object {
        Remove-Item -LiteralPath $_.FullName -Recurse -Force
    }

& (Join-Path $ScriptDir "smoke_check_release.ps1") -Path $DistPath
if ($LASTEXITCODE -ne 0) {
    throw "Release structure smoke check failed."
}

$PreviousSkipUpdate = $env:FP_ANALYSIS_SKIP_UPDATE
try {
    $env:FP_ANALYSIS_SKIP_UPDATE = "1"
    Invoke-Native -FilePath (Join-Path $DistPath "run_fp_analysis_app.exe") `
        -CommandArgs @("--smoke")
} finally {
    $env:FP_ANALYSIS_SKIP_UPDATE = $PreviousSkipUpdate
}

if (Test-Path -LiteralPath $ZipPath) {
    $SafeZipPath = Assert-RepoChildPath -Path $ZipPath
    Remove-Item -LiteralPath $SafeZipPath -Force
}
Compress-Archive -Path $DistPath -DestinationPath $ZipPath

$FreshSmokeRoot = Join-Path $Repo "build\release_smoke"
Remove-RepoBuildPath -Path $FreshSmokeRoot
try {
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $FreshSmokeRoot
    $FreshDistPath = Join-Path $FreshSmokeRoot $PackageFolderName
    if (-not (Test-Path -LiteralPath $FreshDistPath)) {
        throw "Fresh extraction is missing the expected app folder: $FreshDistPath"
    }

    $PreviousSkipUpdate = $env:FP_ANALYSIS_SKIP_UPDATE
    try {
        $env:FP_ANALYSIS_SKIP_UPDATE = "1"
        Invoke-Native -FilePath (Join-Path $FreshDistPath "unblock_app.cmd") `
            -CommandArgs @("--smoke")
    } finally {
        $env:FP_ANALYSIS_SKIP_UPDATE = $PreviousSkipUpdate
    }
} finally {
    Remove-RepoBuildPath -Path $FreshSmokeRoot
}

$Hash = Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256
"$($Hash.Hash)  $(Split-Path $ZipPath -Leaf)" |
    Set-Content -LiteralPath "$ZipPath.sha256.txt" -Encoding UTF8

$Freeze = Invoke-NativeCapture -FilePath $BuildPython -CommandArgs @(
    "-m",
    "pip",
    "freeze"
)
$Freeze | Set-Content -LiteralPath "$ZipPath.build_env_requirements.txt" -Encoding UTF8

$Manifest = [ordered]@{
    version = $Version
    package_folder = $PackageFolderName
    kind = "full-windows"
    branch = Invoke-NativeCapture -FilePath "git" -CommandArgs @(
        "branch",
        "--show-current"
    )
    git_commit = Invoke-NativeCapture -FilePath "git" -CommandArgs @(
        "rev-parse",
        "HEAD"
    )
    tracked_worktree_dirty = [bool]$TrackedStatus
    source_export = if ($AllowDirty) { "tracked-worktree-bytes" } else { "git-blob-bytes" }
    generated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    build_python = $BuildPython
    test_python = if ($SkipTests) { $null } else { $TestPython }
    python = Invoke-NativeCapture -FilePath $BuildPython -CommandArgs @(
        "--version"
    )
    pyinstaller = Invoke-NativeCapture -FilePath $BuildPython -CommandArgs @(
        "-m",
        "PyInstaller",
        "--version"
    )
    updater = Invoke-NativeCapture -FilePath $BuildPython -CommandArgs @(
        "-c",
        "from importlib.metadata import version; print(version('desktop-app-source-updater'))"
    )
    artifact = Split-Path $ZipPath -Leaf
    launcher = "unblock_app.cmd"
    build_env_requirements = Split-Path "$ZipPath.build_env_requirements.txt" -Leaf
    required_paths = @(
        "_internal/",
        "run_fp_analysis_app.exe",
        "unblock_app.cmd",
        "fp_analysis_app/"
    )
    sha256 = $Hash.Hash
}

$Manifest | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath "$ZipPath.manifest.json" -Encoding UTF8

Write-Host "Built full app zip: $ZipPath"
Write-Host "SHA256: $($Hash.Hash)"
