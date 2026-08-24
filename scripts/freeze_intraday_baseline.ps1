[CmdletBinding()]
param(
    [string]$DataDir = "",
    [string]$ReportDir = "",
    [string[]]$Tickers = @(
        "SPY", "QQQ", "DIA", "EEM", "AAPL", "AMZN",
        "META", "TSLA", "SPX", "NDX", "VXX"
    )
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($DataDir)) {
    $DataDir = Join-Path $projectRoot "data\firstrate"
}
if ([string]::IsNullOrWhiteSpace($ReportDir)) {
    $ReportDir = Join-Path $projectRoot "reports\v15_complete"
}

$pythonPath = Join-Path $projectRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python del venv non trovato: $pythonPath"
}
if (-not (Test-Path -LiteralPath $DataDir -PathType Container)) {
    throw "Cartella dati non trovata: $DataDir"
}
if (-not (Test-Path -LiteralPath $ReportDir -PathType Container)) {
    throw "Cartella report non trovata: $ReportDir"
}

$DataDir = (Resolve-Path -LiteralPath $DataDir).Path
$ReportDir = (Resolve-Path -LiteralPath $ReportDir).Path
Set-Location $projectRoot

Write-Host "Validazione integrita' archivi ZIP..."
foreach ($ticker in $Tickers) {
    $archiveMatches = @(
        Get-ChildItem -LiteralPath $DataDir -File |
            Where-Object { $_.Name -like "$($ticker)_1min*.zip" }
    )
    if ($archiveMatches.Count -ne 1) {
        throw "$($ticker): atteso un solo archivio, trovati $($archiveMatches.Count)."
    }
    & $pythonPath -m adaptive.run_zip_integrity_check `
        --file $archiveMatches[0].FullName
    if ($LASTEXITCODE -ne 0) {
        throw "$($ticker): archivio ZIP non valido."
    }
}

$freezeArguments = @(
    "-m", "adaptive.run_research_freeze",
    "--data-dir", $DataDir,
    "--report-dir", $ReportDir,
    "--tickers"
) + $Tickers

& $pythonPath @freezeArguments
if ($LASTEXITCODE -ne 0) {
    throw "Creazione del freeze non riuscita."
}

$freezePath = Join-Path $ReportDir "intraday_research_freeze.json"
if (-not (Test-Path -LiteralPath $freezePath -PathType Leaf)) {
    throw "Freeze atteso ma non trovato: $freezePath"
}

$readinessArguments = @(
    "-m", "adaptive.run_forward_research_readiness",
    "--baseline-freeze", $freezePath,
    "--candidate-freeze", $freezePath
)

& $pythonPath @readinessArguments
if ($LASTEXITCODE -ne 0) {
    throw "Autocontrollo del freeze non riuscito."
}

Write-Host "Baseline descrittiva congelata e verificata: $freezePath"
Write-Host "L'esito iniziale INCONCLUSIVE con zero nuove sedute e' quello previsto."
Write-Host "Nessun segnale, ordine, outcome futuro o P&L e' stato calcolato."
