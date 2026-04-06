param(
    [int]$RandomRuns = 1,
    [int]$LlmRuns = 3,
    [int]$TimeoutSeconds = 1200,
    [string]$ApiBaseUrl = "https://api.openai.com/v1",
    [string]$ModelName = "nvidia/Nemotron-3-Super-49B-v1",
    [string]$OutputJson = "baseline_nemotron_report.json"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ($LlmRuns -gt 0 -and -not $env:OPENAI_API_KEY -and -not $env:HF_TOKEN) {
    Write-Error "Set OPENAI_API_KEY or HF_TOKEN before running this script."
}

$env:API_BASE_URL = $ApiBaseUrl
$env:MODEL_NAME = $ModelName

Write-Host "Running baseline matrix in $repoRoot"
Write-Host "API_BASE_URL=$($env:API_BASE_URL)"
Write-Host "MODEL_NAME=$($env:MODEL_NAME)"
Write-Host "RandomRuns=$RandomRuns LlmRuns=$LlmRuns TimeoutSeconds=$TimeoutSeconds"

$candidatePython = @(
    (Join-Path $repoRoot ".venv/Scripts/python.exe"),
    (Join-Path (Split-Path -Parent $repoRoot) ".venv/Scripts/python.exe")
)

$python = $null
foreach ($candidate in $candidatePython) {
    if (Test-Path $candidate) {
        $python = $candidate
        break
    }
}
if (-not $python) {
    $python = "python"
}

& $python scripts/run_baseline_matrix.py `
    --random-runs $RandomRuns `
    --llm-runs $LlmRuns `
    --timeout-seconds $TimeoutSeconds `
    --output-json $OutputJson

if ($LASTEXITCODE -ne 0) {
    Write-Error "Baseline matrix run failed"
}

Write-Host "Done. Report written to $OutputJson"
