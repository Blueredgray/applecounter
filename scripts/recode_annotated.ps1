# Перекодировать annotated.mp4 (fourcc mp4v -> h264 CRF23), originals удаляются.
param(
    [switch]$KeepOriginal,
    [string]$Crf = "23"
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$files = Get-ChildItem "outputs" -Recurse -Filter "annotated.mp4" -File
Write-Output ("found: " + $files.Count)
$before = ($files | Measure-Object Length -Sum).Sum
$i = 0
foreach ($f in $files) {
    $i++
    $tmp = Join-Path $f.DirectoryName "annotated.h264.tmp.mp4"
    if (Test-Path $tmp) { Remove-Item $tmp -Force }
    Write-Output "[$i/$($files.Count)] $($f.Directory.Name) ($([math]::Round($f.Length/1MB)) MB)"
    & ffmpeg -y -loglevel error -i $f.FullName -c:v libx264 -preset veryfast -crf $Crf -pix_fmt yuv420p -movflags +faststart $tmp
    if ($LASTEXITCODE -ne 0) { Write-Warning "ffmpeg failed for $($f.Name)"; Remove-Item $tmp -Force -ErrorAction SilentlyContinue; continue }
    $newLen = (Get-Item $tmp).Length
    if ($newLen -lt $f.Length) {
        if ($KeepOriginal) {
            Rename-Item $f.FullName "annotated.orig.mp4" -Force
            Rename-Item $tmp $f.FullName -Force
        } else {
            Remove-Item $f.FullName -Force
            Rename-Item $tmp $f.FullName -Force
        }
        Write-Output ("    -> " + [math]::Round($newLen/1MB,1) + " MB")
    } else {
        Write-Output "    (новый не меньше старого, оставлен оригинал)"
        Remove-Item $tmp -Force
    }
}
$after = (Get-ChildItem "outputs" -Recurse -Filter "annotated.mp4" -File | Measure-Object Length -Sum).Sum
Write-Output ("done: " + [math]::Round($before/1GB,2) + " GB -> " + [math]::Round($after/1GB,2) + " GB")
