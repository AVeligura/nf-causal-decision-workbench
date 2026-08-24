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
    throw "Сборку необходимо выполнять в 64-разрядной Windows 10/11."
}

$HostPython = (& py -3.12 -c "import sys; assert sys.maxsize > 2**32; print(sys.executable)").Trim()
if ($LASTEXITCODE -ne 0 -or -not $HostPython) {
    throw "Не найден Python 3.12 x64."
}

if (-not (Test-Path $BuildPython)) {
    & $HostPython -m venv $BuildVenv
}

& $BuildPython -m pip install --disable-pip-version-check --upgrade pip wheel setuptools
& $BuildPython -m pip install --disable-pip-version-check --require-hashes --prefer-binary -r (Join-Path $ProjectRoot "requirements-windows.lock")
& $BuildPython -m pip install --disable-pip-version-check --no-deps -e $ProjectRoot
& $BuildPython -m pip install --disable-pip-version-check "Nuitka==4.1.1" "ordered-set>=4.1,<5" "zstandard>=0.23,<1"

if (Test-Path $BuildRoot) {
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $BuildRoot | Out-Null

$NuitkaArgs = @(
    "-m", "nuitka",
    "--mode=standalone",
    "--enable-plugin=pyside6",
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
    "--nofollow-import-to=pytest,hypothesis,ruff,mypy",
    "--assume-yes-for-downloads",
    "--remove-output",
    "--jobs=$env:NUMBER_OF_PROCESSORS",
    (Join-Path $ProjectRoot "packaging\tester_launcher.py")
)
& $BuildPython @NuitkaArgs
if ($LASTEXITCODE -ne 0) {
    throw "Nuitka завершил сборку с кодом $LASTEXITCODE."
}

$NuitkaDistribution = Join-Path $BuildRoot "tester_launcher.dist"
if (-not (Test-Path $NuitkaDistribution)) {
    throw "Не найден выходной каталог Nuitka: $NuitkaDistribution"
}

if (Test-Path $Distribution) {
    Remove-Item -LiteralPath $Distribution -Recurse -Force
}
New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null
Copy-Item -LiteralPath $NuitkaDistribution -Destination $Distribution -Recurse
Copy-Item -LiteralPath (Join-Path $ProjectRoot "packaging\README_TESTERS_RU.md") -Destination (Join-Path $Distribution "README_TESTERS_RU.md")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "LICENSE") -Destination (Join-Path $Distribution "LICENSE")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "THIRD_PARTY_NOTICES.md") -Destination (Join-Path $Distribution "THIRD_PARTY_NOTICES.md")

$Exe = Join-Path $Distribution "NF_Causal_Workbench.exe"
& $Exe --smoke-check
if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $Distribution "smoke_check.json"))) {
    throw "Собранное приложение не прошло автономную smoke-проверку."
}

$VerificationReport = Join-Path $Distribution "DISTRIBUTION_VERIFICATION.json"
& $BuildPython (Join-Path $ProjectRoot "packaging\verify_tester_distribution.py") $Distribution --report $VerificationReport

Get-ChildItem -LiteralPath $Distribution -File -Recurse | Sort-Object FullName | ForEach-Object {
    $Relative = $_.FullName.Substring($Distribution.Length + 1).Replace("\", "/")
    $Digest = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
    "$Digest  $Relative"
} | Set-Content -LiteralPath (Join-Path $Distribution "FILES_SHA256.txt") -Encoding utf8

if (Test-Path $Archive) {
    Remove-Item -LiteralPath $Archive -Force
}
Compress-Archive -LiteralPath $Distribution -DestinationPath $Archive -CompressionLevel Optimal
$ArchiveDigest = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
"$ArchiveDigest  $([System.IO.Path]::GetFileName($Archive))" | Set-Content -LiteralPath $HashFile -Encoding ascii

if (Test-Path $DataDistribution) {
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

Get-ChildItem -LiteralPath $DataDistribution -File -Recurse | Sort-Object FullName | ForEach-Object {
    $Relative = $_.FullName.Substring($DataDistribution.Length + 1).Replace("\", "/")
    $Digest = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
    "$Digest  $Relative"
} | Set-Content -LiteralPath (Join-Path $DataDistribution "DATA_MANIFEST_SHA256.txt") -Encoding utf8

if (Test-Path $DataArchive) {
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

Write-Host "Сборка завершена:"
Write-Host $Archive
Write-Host "SHA-256: $ArchiveDigest"
Write-Host $DataArchive
Write-Host "SHA-256: $DataArchiveDigest"
