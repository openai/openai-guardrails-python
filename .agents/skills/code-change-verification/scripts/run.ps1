Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = $null

try {
    $repoRoot = (& git -C $scriptDir rev-parse --show-toplevel 2>$null)
} catch {
    $repoRoot = $null
}

if (-not $repoRoot) {
    $repoRoot = (Resolve-Path (Join-Path $scriptDir "..\\..\\..\\..")).Path
} else {
    $repoRoot = ([string]$repoRoot).Trim()
}

Set-Location $repoRoot

function Invoke-VerificationStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )

    Write-Host "Running $Name..."
    & $FilePath @ArgumentList

    if ($LASTEXITCODE -ne 0) {
        throw "code-change-verification: $Name failed with exit code $LASTEXITCODE."
    }

    Write-Host "$Name passed."
}

Invoke-VerificationStep -Name "make format" -FilePath "make" -ArgumentList @("format")
Invoke-VerificationStep -Name "make lint" -FilePath "make" -ArgumentList @("lint")
Invoke-VerificationStep -Name "uv run mypy src tests" -FilePath "uv" -ArgumentList @(
    "run",
    "mypy",
    "src",
    "tests"
)
Invoke-VerificationStep -Name "uv run pyright" -FilePath "uv" -ArgumentList @(
    "run",
    "pyright"
)
Invoke-VerificationStep -Name "make tests" -FilePath "make" -ArgumentList @("tests")

Write-Host "code-change-verification: all commands passed."
