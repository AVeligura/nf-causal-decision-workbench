$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildRoot = Join-Path $ProjectRoot "build\windows_tester"
$ReleaseRoot = Join-Path $ProjectRoot "release"
$ReleaseName = "NF_Causal_Workbench_V3_1_1_Windows_Tester"
$Distribution = Join-Path $ReleaseRoot $ReleaseName
$Archive = Join-Path $ReleaseRoot ($ReleaseName + ".zip")
$HashFile = Join-Path $ReleaseRoot ($ReleaseName + "_SHA256.txt")
$DataReleaseName = "NF_Causal_Workbench_V3_1_1_Test_Data"
$DataDistribution = Join-Path $ReleaseRoot $DataReleaseName
$DataArchive = Join-Path $ReleaseRoot ($DataReleaseName + ".zip")
$DataHashFile = Join-Path $ReleaseRoot ($DataReleaseName + "_SHA256.txt")
$GeneratedReleaseNotes = Join-Path $ReleaseRoot "RELEASE_NOTES_GENERATED.md"
$BuildVenv = Join-Path $ProjectRoot ".build_venv"
$BuildPython = Join-Path $BuildVenv "Scripts\python.exe"

if ([System.Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
    throw "The build must run on 64-bit Windows 10 or Windows 11."
}

$ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw "Project Python was not found: $ProjectPython"
}

$HostPythonOutput = & $ProjectPython -c "import sys; assert sys.version_info[:2] == (3, 12); assert sys.maxsize > 2**32; print(sys._base_executable)"

if ($LASTEXITCODE -ne 0 -or -not $HostPythonOutput) {
    throw "The project .venv must use 64-bit Python 3.12."
}

$HostPython = ([string]$HostPythonOutput).Trim()

if (-not (Test-Path -LiteralPath $HostPython)) {
    throw "Base Python was not found: $HostPython"
}

if (-not (Test-Path -LiteralPath $BuildPython)) {
    & $HostPython -m venv $BuildVenv
}

& $BuildPython -m pip install --disable-pip-version-check --upgrade pip wheel setuptools
& $BuildPython -m pip install --disable-pip-version-check --require-hashes --prefer-binary -r (Join-Path $ProjectRoot "requirements-windows.lock")
& $BuildPython -m pip install --disable-pip-version-check --no-deps -e $ProjectRoot
& $BuildPython -m pip install --disable-pip-version-check "Nuitka==4.1.1" "ordered-set>=4.1,<5" "zstandard>=0.23,<1"

if (Test-Path -LiteralPath $BuildRoot) {
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $BuildRoot | Out-Null

$NuitkaArgs = @(
    "-m", "nuitka",
    "--mode=standalone",
    "--enable-plugin=pyside6",
    "--msvc=latest",
    "--module-parameter=torch-disable-jit=yes",
    "--windows-console-mode=disable",
    "--windows-company-name=NF-Causal Lab",
    "--windows-product-name=NF-Causal Decision Workbench",
    "--windows-file-version=3.1.1.0",
    "--windows-product-version=3.1.1.0",
    "--output-filename=NF_Causal_Workbench.exe",
    "--output-dir=$BuildRoot",
    "--report=$(Join-Path $BuildRoot 'nuitka-build-report.xml')",
    "--include-data-dir=$(Join-Path $ProjectRoot 'data')=data",
    "--include-data-dir=$(Join-Path $ProjectRoot 'configs')=configs",
    "--include-package=domain",
    "--include-package=engine",
    "--include-package=study",
    "--include-package=runtime",
    "--include-package=visualization",
    "--include-package=ui",
    "--include-package=scipy._external.array_api_compat.numpy",
    "--include-module=torch.testing._internal.logging_tensor",
    "--nofollow-import-to=pytest,hypothesis,ruff,mypy,torch.testing._internal.common_methods_invocations",
    "--assume-yes-for-downloads",
    "--remove-output",
    "--low-memory",
    (Join-Path $ProjectRoot "packaging\tester_launcher.py")
)

& $BuildPython @NuitkaArgs
if ($LASTEXITCODE -ne 0) {
    throw "Nuitka failed with exit code $LASTEXITCODE."
}

$NuitkaDistribution = Join-Path $BuildRoot "tester_launcher.dist"
if (-not (Test-Path -LiteralPath $NuitkaDistribution)) {
    throw "Nuitka output directory was not found: $NuitkaDistribution"
}

if (Test-Path -LiteralPath $Distribution) {
    Remove-Item -LiteralPath $Distribution -Recurse -Force
}
New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null
Copy-Item -LiteralPath $NuitkaDistribution -Destination $Distribution -Recurse
Copy-Item -LiteralPath (Join-Path $ProjectRoot "packaging\README_TESTERS_RU.md") -Destination (Join-Path $Distribution "README_TESTERS_RU.md")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "LICENSE") -Destination (Join-Path $Distribution "LICENSE")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "THIRD_PARTY_NOTICES.md") -Destination (Join-Path $Distribution "THIRD_PARTY_NOTICES.md")

$Exe = Join-Path $Distribution "NF_Causal_Workbench.exe"
$SmokeFile = Join-Path $Distribution "smoke_check.json"

# Исключаем ложноположительный результат от предыдущего запуска.
Remove-Item -LiteralPath $SmokeFile -Force -ErrorAction SilentlyContinue

Write-Host "Запуск автономной smoke-проверки..."

$SmokeProcess = Start-Process `
    -FilePath $Exe `
    -ArgumentList "--smoke-check" `
    -Wait `
    -PassThru

$SmokeExitCode = [int32]$SmokeProcess.ExitCode
$SmokeExitUInt32 = [System.BitConverter]::ToUInt32(
    [System.BitConverter]::GetBytes($SmokeExitCode), 0
)
$SmokeExitHex = "0x{0:X8}" -f $SmokeExitUInt32

$SmokeExists = Test-Path -LiteralPath $SmokeFile
$SmokePassed = $false
$GuiSmokeStatus = $null
$GuiWorkspaceCount = $null

if ($SmokeExists) {
    try {
        $SmokeResult = Get-Content -LiteralPath $SmokeFile -Raw | ConvertFrom-Json
        $GuiSmokeStatus = $SmokeResult.gui_main_window
        $GuiWorkspaceCount = $SmokeResult.gui_workspace_count
        $SmokePassed = (
            $SmokeResult.status -eq "PASS" -and
            $GuiSmokeStatus -eq "PASS" -and
            [int]$GuiWorkspaceCount -eq 5
        )
    }
    catch {
        Write-Warning "smoke_check.json существует, но не является корректным JSON:"
        Write-Warning $_.Exception.Message
    }
}

if ($SmokeExitCode -ne 0 -or -not $SmokeExists -or -not $SmokePassed) {
    Write-Host ""
    Write-Host "===== SMOKE CHECK DIAGNOSTICS ====="
    Write-Host "Exit code: $SmokeExitCode"
    Write-Host "Exit code HEX: $SmokeExitHex"
    Write-Host "smoke_check.json exists: $SmokeExists"
    Write-Host "GUI main window: $GuiSmokeStatus"
    Write-Host "GUI workspace count: $GuiWorkspaceCount"

    $DiagnosticFiles = Get-ChildItem `
        -LiteralPath $ReleaseRoot, $BuildRoot `
        -Recurse `
        -File `
        -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -in @("startup_error.log", "smoke_check.json")
        } |
        Sort-Object LastWriteTime -Descending

    if ($DiagnosticFiles) {
        Write-Host ""
        Write-Host "Diagnostic files:"
        $DiagnosticFiles |
            Select-Object FullName, Length, LastWriteTime |
            Format-Table -AutoSize |
            Out-Host

        foreach ($DiagnosticFile in $DiagnosticFiles) {
            Write-Host ""
            Write-Host "----- $($DiagnosticFile.FullName) -----"
            Get-Content -LiteralPath $DiagnosticFile.FullName -ErrorAction SilentlyContinue |
                Out-Host
        }
    }
    else {
        Write-Host "No startup_error.log or smoke_check.json files found."
    }

    Write-Host "==================================="
    throw "Собранное приложение не прошло автономную smoke-проверку."
}

Write-Host "Автономная smoke-проверка: PASS"
Write-Host "GUI main window: $GuiSmokeStatus"
Write-Host "GUI workspace count: $GuiWorkspaceCount"
Write-Host "Exit code: $SmokeExitCode"

$VerificationReport = Join-Path $Distribution "DISTRIBUTION_VERIFICATION.json"
& $BuildPython (Join-Path $ProjectRoot "packaging\verify_tester_distribution.py") $Distribution --report $VerificationReport
if ($LASTEXITCODE -ne 0) {
    throw "Distribution verification failed with exit code $LASTEXITCODE."
}

Get-ChildItem -LiteralPath $Distribution -File -Recurse | Sort-Object FullName | ForEach-Object {
    $Relative = $_.FullName.Substring($Distribution.Length + 1).Replace("\", "/")
    $Digest = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
    "$Digest  $Relative"
} | Set-Content -LiteralPath (Join-Path $Distribution "FILES_SHA256.txt") -Encoding utf8

if (Test-Path -LiteralPath $Archive) {
    Remove-Item -LiteralPath $Archive -Force
}
Compress-Archive -LiteralPath $Distribution -DestinationPath $Archive -CompressionLevel Optimal
$ArchiveDigest = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
"$ArchiveDigest  $([System.IO.Path]::GetFileName($Archive))" | Set-Content -LiteralPath $HashFile -Encoding ascii

if (Test-Path -LiteralPath $DataDistribution) {
    Remove-Item -LiteralPath $DataDistribution -Recurse -Force
}
New-Item -ItemType Directory -Path (Join-Path $DataDistribution "data\raw\uci_polish") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $DataDistribution "configs") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $DataDistribution "examples") -Force | Out-Null

Copy-Item -LiteralPath (Join-Path $ProjectRoot "data\raw\uci_polish\data.csv") -Destination (Join-Path $DataDistribution "data\raw\uci_polish\data.csv")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "data\raw\uci_polish\ATTRIBUTION.md") -Destination (Join-Path $DataDistribution "data\raw\uci_polish\ATTRIBUTION.md")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "configs\reference.yaml") -Destination (Join-Path $DataDistribution "configs\reference.yaml")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "packaging\README_DATA_RU.md") -Destination (Join-Path $DataDistribution "README_DATA_RU.md")
& $BuildPython (Join-Path $ProjectRoot "packaging\generate_release_sample.py") --output (Join-Path $DataDistribution "examples\sample_import_reference_seed_20260814.csv")
if ($LASTEXITCODE -ne 0) {
    throw "Release sample generation failed with exit code $LASTEXITCODE."
}

Get-ChildItem -LiteralPath $DataDistribution -File -Recurse | Sort-Object FullName | ForEach-Object {
    $Relative = $_.FullName.Substring($DataDistribution.Length + 1).Replace("\", "/")
    $Digest = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
    "$Digest  $Relative"
} | Set-Content -LiteralPath (Join-Path $DataDistribution "DATA_MANIFEST_SHA256.txt") -Encoding utf8

if (Test-Path -LiteralPath $DataArchive) {
    Remove-Item -LiteralPath $DataArchive -Force
}
Compress-Archive -LiteralPath $DataDistribution -DestinationPath $DataArchive -CompressionLevel Optimal
$DataArchiveDigest = (Get-FileHash -Algorithm SHA256 -LiteralPath $DataArchive).Hash.ToLowerInvariant()
"$DataArchiveDigest  $([System.IO.Path]::GetFileName($DataArchive))" | Set-Content -LiteralPath $DataHashFile -Encoding ascii

$ReleaseNotes = Get-Content -LiteralPath (Join-Path $ProjectRoot "RELEASE_NOTES.md") -Raw
$ReleaseNotes += "`r`n## SHA-256`r`n`r`n"
$ReleaseNotes += "- ``$([System.IO.Path]::GetFileName($Archive))``: ``$ArchiveDigest```r`n"
$ReleaseNotes += "- ``$([System.IO.Path]::GetFileName($DataArchive))``: ``$DataArchiveDigest```r`n"
Set-Content -LiteralPath $GeneratedReleaseNotes -Value $ReleaseNotes -Encoding utf8

Write-Host "Build completed:"
Write-Host $Archive
Write-Host "SHA-256: $ArchiveDigest"
Write-Host $DataArchive
Write-Host "SHA-256: $DataArchiveDigest"
