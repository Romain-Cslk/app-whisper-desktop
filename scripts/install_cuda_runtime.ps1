[CmdletBinding()]
param([string]$ProjectRoot = "")

$ErrorActionPreference = 'Stop'
if (-not $ProjectRoot) {
    $ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
} else {
    $ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
}
Set-Location -LiteralPath $ProjectRoot

if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem) {
    throw 'Le runtime GPU de ce projet cible Windows x64.'
}

$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Environnement .venv absent.'
}

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) { throw 'uv est introuvable dans le PATH.' }
$uv = $uvCommand.Source

function Invoke-NativeChecked([string]$Program, [string[]]$Arguments, [string]$FailureMessage) {
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Program @Arguments
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($code -ne 0) { throw "$FailureMessage (code $code)" }
}

Write-Host '1/5 - Resolution du lock avec la pile CUDA 12.8 epinglee...'
Invoke-NativeChecked $uv @('lock') 'uv lock a echoue.'

Write-Host '2/5 - Synchronisation exacte du .venv...'
Invoke-NativeChecked $uv @('sync', '--locked', '--group', 'dev', '--group', 'build', '--python', '3.11') `
    'uv sync a echoue.'

Write-Host '3/5 - Verification des versions et DLL CUDA...'
Invoke-NativeChecked $python @('scripts/verify_cuda_runtime.py') `
    'La pile CUDA installee n est pas coherente.'

$models = Join-Path $ProjectRoot 'src\transcripteur_whisper\assets\diarization'
$segmentation = Join-Path $models 'segmentation.onnx'
$embedding = Join-Path $models 'embedding.onnx'
if (-not (Test-Path -LiteralPath $segmentation) -or -not (Test-Path -LiteralPath $embedding)) {
    Write-Host '4/5 - Preparation des modeles de diarisation...'
    Invoke-NativeChecked $python @('scripts/prepare_diarization_assets.py', $models) `
        'Preparation des modeles de diarisation echouee.'
} else {
    Write-Host '4/5 - Modeles de diarisation deja presents.'
}

Write-Host '5/5 - Smoke test REEL du worker de diarisation CUDA...'
$smoke = Join-Path $ProjectRoot 'build\diarization-cuda-smoke-v102'
if (Test-Path -LiteralPath $smoke) { Remove-Item -LiteralPath $smoke -Recurse -Force }
Invoke-NativeChecked $python @(
    'scripts/validate_diarization.py',
    '--models-dir', $models,
    '--device', 'cuda',
    '--work-dir', $smoke
) 'Le worker sherpa CUDA ne parvient pas a charger/executer les modeles.'

Write-Host ''
Write-Host 'RUNTIME CUDA 12.8 : VALIDE'
Write-Host 'SHERPA CUDA : VALIDE PAR INFERENCE REELLE'
Write-Host 'Le .venv est maintenant reproductible par uv sync.'
