# Re-run per-tree assignment for all finished VFC runs (no GPU needed).
# Uses the updated trees defaults (in-row spacing 1.0 m, apple 70 mm).
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$env:PYTHONPATH = "$root\src"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $root

$runs = Get-ChildItem -Path "$root\outputs" -Directory | Where-Object {
    (Test-Path "$($_.FullName)\tracks_frame.csv") -and (Test-Path "$($_.FullName)\per_frame.csv")
}
# только прогоны с посегментной картой деревьев (VFC); MinneApple/синтетика не трогаются
$runs = $runs | Where-Object {
    $sj = "$($_.FullName)\summary.json"
    if (-not (Test-Path $sj)) { return $false }
    $s = Get-Content $sj -Raw | ConvertFrom-Json
    [string]$s.tree_method -like "segments*"
}
$i = 0
foreach ($r in $runs) {
    $i++
    Write-Output "[$i/$($runs.Count)] $($r.Name)"
    python -m applecounter retree --run "$($r.FullName)"
    if ($LASTEXITCODE -ne 0) { Write-Output "  FAILED ($LASTEXITCODE)" }
}
Write-Output "done: $($runs.Count) runs"
