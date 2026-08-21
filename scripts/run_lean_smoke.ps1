[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$manifestPath = Join-Path $repoRoot "lean\runtime.json"
$algorithmPath = (Resolve-Path (Join-Path $repoRoot "lean\smoke")).Path

if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Manifest LEAN non trovato: $manifestPath"
}

$runtime = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$requiredProperties = @("image", "digest", "mode", "live_enabled")
foreach ($propertyName in $requiredProperties) {
    if ($null -eq $runtime.PSObject.Properties[$propertyName]) {
        throw "Proprieta obbligatoria mancante nel manifest: $propertyName"
    }
}

if ([string]$runtime.mode -ne "backtesting") {
    throw "Modalita non consentita: il runner accetta solo backtesting."
}

if ([bool]$runtime.live_enabled) {
    throw "Configurazione rifiutata: live_enabled deve restare false."
}

$image = [string]$runtime.image
$expectedDigest = [string]$runtime.digest
if ($expectedDigest -notmatch "^sha256:[0-9a-f]{64}$") {
    throw "Digest LEAN non valido nel manifest."
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker non e disponibile nel PATH. Avvia Docker Desktop e riprova."
}

& docker info --format "{{.ServerVersion}}" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop non risponde. Avvialo e attendi che il motore sia pronto."
}

$repoDigests = (& docker image inspect $image --format "{{range .RepoDigests}}{{println .}}{{end}}" 2>$null) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw "Immagine $image non trovata. Esegui: docker pull $image"
}

if ($repoDigests -notmatch [regex]::Escape($expectedDigest)) {
    throw "Il digest locale di $image non coincide con quello approvato: $expectedDigest"
}

$runId = "{0}-{1}" -f (Get-Date -Format "yyyyMMdd-HHmmss"), $PID
$resultsPath = Join-Path $repoRoot ".tradingai\lean\results\$runId"
[void](New-Item -ItemType Directory -Force -Path $resultsPath)
$resultsPath = (Resolve-Path $resultsPath).Path

$containerName = "tradingai-lean-$runId"
$projectMount = "type=bind,source=$algorithmPath,target=/LeanCLI,readonly"
$resultsMount = "type=bind,source=$resultsPath,target=/Results"

$dockerArguments = @(
    "run",
    "--rm",
    "--name", $containerName,
    "--stop-signal", "SIGINT",
    "--env", "PYTHONDONTWRITEBYTECODE=1",
    "--mount", $projectMount,
    "--mount", $resultsMount,
    $image,
    "--environment", "backtesting",
    "--algorithm-type-name", "LeanSmokeTestAlgorithm",
    "--algorithm-language", "Python",
    "--algorithm-location", "/LeanCLI/main.py",
    "--data-folder", "/Lean/Data",
    "--results-destination-folder", "/Results",
    "--backtest-name", "TradingAI LEAN smoke test",
    "--algorithm-id", $runId,
    "--close-automatically", "true"
)

Write-Host "Avvio smoke test LEAN locale (solo backtest)..."
& docker @dockerArguments
if ($LASTEXITCODE -ne 0) {
    throw "Lo smoke test LEAN e terminato con codice $LASTEXITCODE."
}

$resultFiles = @(Get-ChildItem -LiteralPath $resultsPath -File -Filter "*.json")
if ($resultFiles.Count -eq 0) {
    throw "LEAN si e chiuso senza produrre risultati JSON in $resultsPath"
}

Write-Host "Smoke test completato. Risultati: $resultsPath"
