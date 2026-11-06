# Batch run of all VideoFruitCounting videos through our pipeline.
# Usage (from project root):
#   powershell -ExecutionPolicy Bypass -File scripts/run_vfc_all.ps1
# Each video: detection (YOLO26s@1280) -> BoT-SORT -> per-stretch count
# from SFRAME + export to VFC format (all_predictions.csv).
# Optional -Only <substring> filter (e.g. -Only "ZED").
#
# IMPORTANT: rotation depends on the camera rig:
#   Azure Kinect videos -> rotate90 = -1 (counter-clockwise)
#   ZED 2 videos        -> rotate90 = +1 (clockwise; camera mounted upside down)
# A run is skipped only if <out>\rot.txt records the same rotation.
# NOTE: ASCII-only on purpose - Windows PowerShell 5.1 misreads UTF-8
# without BOM and Cyrillic breaks parsing.

param(
    [string]$Only = "",
    [string]$Detector = "weights/runs/minneapple26s1280/weights/best.pt",
    [int]$Imgsz = 1280,
    [string]$Tracker = "botsort"
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

# package lives in src/ and is found via PYTHONPATH in the interactive shell;
# a fresh powershell does not inherit it -> set explicitly
$env:PYTHONPATH = "$root\src"
$env:PYTHONIOENCODING = "utf-8"

# runs finished under a custom --out name (rotation recorded in rot.txt below)
$doneMap = @{
    "210928_100624_k_r1_e_015_225_162" = "outputs\vfc_k_r1_e"
}

$videosRoot = "data/raw/videofruit/Video Fruit Counting/Videos"
$sframe = "data/raw/videofruit/Video Fruit Counting/stretch_sframe.csv"

$videos = Get-ChildItem -Recurse "$videosRoot\*.mp4" | Sort-Object FullName
if ($Only) { $videos = $videos | Where-Object { $_.Name -like "*$Only*" } }

Write-Host ("total videos: " + $videos.Count)
$i = 0
foreach ($v in $videos) {
    $i++
    $out = "outputs\vfc_" + $v.BaseName

    # camera-dependent rotation: z_r* = ZED 2 (clockwise), k_r* = Azure Kinect
    if ($v.BaseName -match "_z_r") { $rot = 1 } else { $rot = -1 }

    $skip = $false
    $doneDir = $out
    if (-not (Test-Path "$out\summary.json") -and $doneMap.ContainsKey($v.BaseName)) {
        $doneDir = $doneMap[$v.BaseName]
    }
    if (Test-Path "$doneDir\summary.json") {
        $rotFile = "$doneDir\rot.txt"
        if (Test-Path $rotFile) {
            $skip = ((Get-Content $rotFile -Raw).Trim() -eq "$rot")
        } else {
            # old runs have no marker: Kinect runs used -1 and are correct,
            # ZED runs used -1 too and must be redone with +1
            $skip = ($rot -eq -1)
        }
    }

    if ($skip) {
        Write-Host "[$i/$($videos.Count)] $($v.BaseName) - already done (rot=$rot), skip"
        continue
    }

    if ($doneDir -ne $out -and (Test-Path $doneDir)) {
        Write-Host "[$i/$($videos.Count)] $($v.BaseName) - old run used wrong rotation, removing $doneDir"
        Remove-Item $doneDir -Recurse -Force
    }

    Write-Host "[$i/$($videos.Count)] $($v.BaseName) (rotate90=$rot) ..."
    python -m applecounter run --video "`"$($v.FullName)`"" --out $out `
        --set video.rotate90=$rot `
        --set detector.name=yolo --set detector.yolo.weights=$Detector --set detector.yolo.imgsz=$Imgsz `
        --set trees.method=segments --set trees.segments_file="$sframe" `
        --set tracker.name=$Tracker
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "run $($v.BaseName) failed ($LASTEXITCODE)"
        continue
    }
    Set-Content -Path "$out\rot.txt" -Value "$rot" -Encoding ascii

    # export to their format for assign_apples.py
    python -m applecounter vfc-export --run $out --segments $sframe
}

Write-Host "done. Per-video results: outputs\vfc_*\summary.json"
