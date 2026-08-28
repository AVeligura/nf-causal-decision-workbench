$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ReleaseRoot = Join-Path $ProjectRoot "release"
$Repo = "AVeligura/nf-causal-decision-workbench"
$Tag = "v3.1.1"
$TargetCommit = "91b2189297cd422035f3903d2a580ce5519dc419"
$Title = "NF-Causal Decision Workbench V3.1.1 - Windows testing build"

$TesterZip = Join-Path $ReleaseRoot "NF_Causal_Workbench_V3_1_1_Windows_Tester.zip"
$TesterHashFile = Join-Path $ReleaseRoot "NF_Causal_Workbench_V3_1_1_Windows_Tester_SHA256.txt"
$DataZip = Join-Path $ReleaseRoot "NF_Causal_Workbench_V3_1_1_Test_Data.zip"
$DataHashFile = Join-Path $ReleaseRoot "NF_Causal_Workbench_V3_1_1_Test_Data_SHA256.txt"
$NotesFile = Join-Path $ReleaseRoot "RELEASE_NOTES_GENERATED.md"

$ExpectedTesterSha256 = "f516e016c3f5201627ea3df50de0fe6883395fcd3fdaa812cbce14af32f64112"
$ExpectedDataSha256 = "b2eb872de043d6cb2f6a92d8443766c0816a25c2fdb10ce346c563786fa4f8be"

$RequiredFiles = @(
    $TesterZip,
    $TesterHashFile,
    $DataZip,
    $DataHashFile,
    $NotesFile
)

foreach ($Path in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required release file was not found: $Path"
    }
}

$TesterDigest = (Get-FileHash -Algorithm SHA256 -LiteralPath $TesterZip).Hash.ToLowerInvariant()
$DataDigest = (Get-FileHash -Algorithm SHA256 -LiteralPath $DataZip).Hash.ToLowerInvariant()

if ($TesterDigest -ne $ExpectedTesterSha256) {
    throw "Tester ZIP checksum mismatch. Expected $ExpectedTesterSha256, got $TesterDigest"
}
if ($DataDigest -ne $ExpectedDataSha256) {
    throw "Test Data ZIP checksum mismatch. Expected $ExpectedDataSha256, got $DataDigest"
}

$TesterHashText = Get-Content -LiteralPath $TesterHashFile -Raw
$DataHashText = Get-Content -LiteralPath $DataHashFile -Raw
$NotesText = Get-Content -LiteralPath $NotesFile -Raw

if ($TesterHashText -notmatch [regex]::Escape($ExpectedTesterSha256)) {
    throw "Tester checksum file does not contain the verified digest."
}
if ($DataHashText -notmatch [regex]::Escape($ExpectedDataSha256)) {
    throw "Test Data checksum file does not contain the verified digest."
}
if ($NotesText -notmatch [regex]::Escape($ExpectedTesterSha256) -or
    $NotesText -notmatch [regex]::Escape($ExpectedDataSha256)) {
    throw "Generated release notes do not contain both verified release digests."
}

$Gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $Gh) {
    throw "GitHub CLI (gh) was not found. Install it from https://cli.github.com/ and run 'gh auth login'."
}

& gh auth status --hostname github.com
if ($LASTEXITCODE -ne 0) {
    throw "GitHub CLI is not authenticated. Run 'gh auth login' first."
}

$AssetPaths = @(
    $TesterZip,
    $TesterHashFile,
    $DataZip,
    $DataHashFile
)

# Windows PowerShell 5.1 can turn stderr from a native command into a
# terminating NativeCommandError when ErrorActionPreference is Stop.
# Probe release existence through cmd.exe so a normal "not found" result
# is represented only by the process exit code.
& cmd.exe /d /c "gh release view $Tag --repo $Repo >nul 2>nul"
$ReleaseExists = ($LASTEXITCODE -eq 0)

if ($ReleaseExists) {
    Write-Host "Release $Tag already exists; replacing assets with the verified local build."
    $UploadArgs = @("release", "upload", $Tag) + $AssetPaths + @("--repo", $Repo, "--clobber")
    & gh @UploadArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upload verified release assets."
    }

    $EditArgs = @(
        "release", "edit", $Tag,
        "--repo", $Repo,
        "--title", $Title,
        "--notes-file", $NotesFile,
        "--prerelease"
    )
    & gh @EditArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to update release metadata."
    }
}
else {
    Write-Host "Creating prerelease $Tag from verified local assets."
    $CreateArgs = @("release", "create", $Tag) + $AssetPaths + @(
        "--repo", $Repo,
        "--target", $TargetCommit,
        "--title", $Title,
        "--notes-file", $NotesFile,
        "--prerelease"
    )
    & gh @CreateArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create GitHub prerelease."
    }
}

Write-Host ""
Write-Host "Published verified V3.1.1 release assets:"
Write-Host "Tester ZIP SHA-256:    $TesterDigest"
Write-Host "Test Data ZIP SHA-256: $DataDigest"
& gh release view $Tag --repo $Repo --json url,tagName,name,isPrerelease --jq '.url'
