[CmdletBinding()]
param([switch]$BuildInstaller)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $root
if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem) {
    throw 'This release validation requires Windows x64.'
}
function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}
$uv = (Get-Command uv -ErrorAction Stop).Source
Invoke-Checked $uv @('sync', '--locked', '--group', 'dev', '--group', 'build', '--python', '3.11')
$python = Join-Path $root '.venv\Scripts\python.exe'
Invoke-Checked $python @('-c', 'import sys; assert sys.version_info[:2]==(3,11); assert sys.maxsize>2**32')
Invoke-Checked $python @('-m', 'ruff', 'check', '.')
Invoke-Checked $python @('-m', 'compileall', '-q', 'src', 'tests', 'scripts', 'packaging')
$reportDir = Join-Path $root 'build\diarization-validation'
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
$previousQt = $env:QT_QPA_PLATFORM
$env:QT_QPA_PLATFORM = 'offscreen'
try {
    $junit = Join-Path $reportDir 'pytest.xml'
    Invoke-Checked $python @('-m', 'pytest', '-q', ('--junitxml=' + $junit))
    Invoke-Checked $python @('scripts/check_diarization_validation.py', $junit)
    $assets = Join-Path $root 'src\transcripteur_whisper\assets\diarization'
    Invoke-Checked $python @('scripts/prepare_diarization_assets.py', $assets)
    Invoke-Checked $python @('-m', 'transcripteur_whisper', '--smoke-test', '--smoke-report', (Join-Path $reportDir 'qt-source.json'))
    Invoke-Checked $python @('scripts/validate_diarization.py', '--models-dir', $assets,
                            '--work-dir', (Join-Path $reportDir 'worker'))
} finally {
    $env:QT_QPA_PLATFORM = $previousQt
}
if ($BuildInstaller) {
    & (Join-Path $PSScriptRoot 'build_windows.ps1') -RequireInstaller
    if ($LASTEXITCODE -ne 0) { throw 'Windows installer build failed.' }
}
Write-Host 'Automated Windows gates passed. This is not an acoustic quality benchmark.'
Write-Host 'Before release, follow the real-meeting acceptance checks in docs/DIARIZATION.md.'
